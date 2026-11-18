// Full-system simulation of solo_core: real firmware, real accelerator RTL,
// a 1 GiB DDR model preloaded with the compact model, the runtime scratch
// image and the layer-0 sum table, and a bit-level UART host.
//
//   tb_solo MODEL SCRATCH SUM_TABLE PROMPT_IDS GENERATE
//
// Prints each generated token with its per-phase cycle profile.
#include <sys/mman.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#include "Vsolo_core.h"
#include "verilated.h"

static constexpr uint64_t DDR_BYTES = 1ull << 30;
static constexpr uint32_t CLKS_PER_BIT = 67;  // 200 MHz / 3 Mbaud
static constexpr int READ_LATENCY = 24;       // UI cycles to first beat
static constexpr int WRITE_LATENCY = 12;

static uint8_t *ddr;
static Vsolo_core *top;
static uint64_t cycle;

static void load(const char *path, uint64_t offset) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  size_t n = fread(ddr + offset, 1, DDR_BYTES - offset, f);
  fclose(f);
  fprintf(stderr, "loaded %s: %zu bytes at DDR+0x%08llx\n", path, n,
          (unsigned long long)offset);
}

// ------------------------------------------------------------- AXI model
struct Burst { uint64_t address; uint32_t beats; uint32_t id; uint64_t ready_at; };
static std::deque<Burst> reads, writes;
static std::deque<std::pair<uint32_t, uint64_t>> responses;  // id, ready_at
static uint32_t read_beat;
static uint32_t write_beat;

static void axi_inputs() {
  top->m_axi_arready = reads.size() < 8;
  top->m_axi_awready = writes.size() < 8;
  top->m_axi_wready = !writes.empty();
  top->m_axi_rvalid = 0;
  if (!reads.empty() && cycle >= reads.front().ready_at) {
    const Burst &b = reads.front();
    uint64_t address = (b.address & ~31ull) + 32ull * read_beat;
    memcpy(top->m_axi_rdata.data(), ddr + (address & (DDR_BYTES - 1)), 32);
    top->m_axi_rvalid = 1;
    top->m_axi_rid = b.id;
    top->m_axi_rresp = 0;
    top->m_axi_rlast = read_beat + 1 == b.beats;
  }
  top->m_axi_bvalid = 0;
  if (!responses.empty() && cycle >= responses.front().second) {
    top->m_axi_bvalid = 1;
    top->m_axi_bid = responses.front().first;
    top->m_axi_bresp = 0;
  }
}

// Sample handshakes on the rising edge (values settled before the edge).
static void axi_handshakes() {
  if (top->m_axi_arvalid && top->m_axi_arready)
    reads.push_back({top->m_axi_araddr, top->m_axi_arlen + 1u,
                     top->m_axi_arid, cycle + READ_LATENCY});
  if (top->m_axi_awvalid && top->m_axi_awready)
    writes.push_back({top->m_axi_awaddr, top->m_axi_awlen + 1u,
                      top->m_axi_awid, 0});
  if (top->m_axi_wvalid && top->m_axi_wready) {
    Burst &b = writes.front();
    uint64_t address = ((b.address & ~31ull) + 32ull * write_beat) &
                       (DDR_BYTES - 1);
    const uint32_t *data = top->m_axi_wdata.data();
    uint32_t strobe = top->m_axi_wstrb;
    for (int i = 0; i < 32; ++i)
      if (strobe & (1u << i))
        ddr[address + i] = (uint8_t)(data[i / 4] >> (8 * (i % 4)));
    if (++write_beat == b.beats) {
      responses.push_back({b.id, cycle + WRITE_LATENCY});
      writes.pop_front();
      write_beat = 0;
    }
  }
  if (top->m_axi_rvalid && top->m_axi_rready) {
    if (++read_beat == reads.front().beats) {
      reads.pop_front();
      read_beat = 0;
    }
  }
  if (top->m_axi_bvalid && top->m_axi_bready) responses.pop_front();
}

// ------------------------------------------------------------ UART host
static std::deque<uint8_t> uart_out;  // host -> board
static std::deque<uint8_t> uart_in;   // board -> host
static uint32_t tx_counter, tx_bit;   // host transmitter state
static uint16_t tx_shift;
static uint32_t rx_counter, rx_bit;   // host receiver state
static uint8_t rx_shift;
static bool rx_active;
static uint8_t last_tx = 1;

static void uart_step() {
  // Host transmit, LSB first, 1 start + 8 data + 1 stop bit.
  if (tx_bit == 0 && !uart_out.empty()) {
    tx_shift = (uint16_t)((uart_out.front() << 1) | 0x200);
    uart_out.pop_front();
    tx_bit = 10;
    tx_counter = 0;
  }
  if (tx_bit) {
    top->rx = tx_shift & 1;
    if (++tx_counter == CLKS_PER_BIT) {
      tx_counter = 0;
      tx_shift >>= 1;
      --tx_bit;
    }
  } else {
    top->rx = 1;
  }
  // Host receive: detect the start edge, sample mid-bit.
  uint8_t line = top->tx;
  if (!rx_active) {
    if (last_tx == 1 && line == 0) {
      rx_active = true;
      rx_counter = 0;
      rx_bit = 0;
      rx_shift = 0;
    }
  } else if (++rx_counter == CLKS_PER_BIT / 2 + CLKS_PER_BIT * rx_bit) {
    if (rx_bit >= 1 && rx_bit <= 8) rx_shift |= (uint8_t)(line << (rx_bit - 1));
    if (rx_bit == 9) {
      uart_in.push_back(rx_shift);
      rx_active = false;
    }
    ++rx_bit;
  }
  last_tx = line;
}

// ------------------------------------------------------------- clocking
// 0.5 ns steps: clk (200 MHz) every 10 steps, fp_clk (125 MHz) every 16.
static uint64_t step;
static void tick() {
  for (;;) {
    ++step;
    bool clk_edge = step % 5 == 0;
    bool fp_edge = step % 8 == 0;
    if (!clk_edge && !fp_edge) continue;
    if (fp_edge) top->fp_clk_i = !top->fp_clk_i;
    if (clk_edge) {
      if (!top->clk_i) {  // rising edge next
        axi_handshakes();
        top->clk_i = 1;
        top->eval();
        ++cycle;
        axi_inputs();
        uart_step();
        top->eval();
        return;
      }
      top->clk_i = 0;
    }
    top->eval();
  }
}

static void run(uint64_t cycles) { for (uint64_t i = 0; i < cycles; ++i) tick(); }

static uint8_t get_byte(uint64_t timeout = 2000000000ull) {
  uint64_t deadline = cycle + timeout;
  while (uart_in.empty()) {
    tick();
    if (cycle > deadline) {
      fprintf(stderr, "UART timeout at cycle %llu\n", (unsigned long long)cycle);
      exit(1);
    }
  }
  uint8_t b = uart_in.front();
  uart_in.pop_front();
  return b;
}

static uint32_t get_u32() {
  uint32_t v = 0;
  for (int i = 0; i < 4; ++i) v |= (uint32_t)get_byte() << (8 * i);
  return v;
}

static void put_u32(uint32_t v) {
  for (int i = 0; i < 4; ++i) uart_out.push_back((uint8_t)(v >> (8 * i)));
}

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: %s MODEL SCRATCH SUM_TABLE PROMPT_IDS GENERATE\n",
            argv[0]);
    return 2;
  }
  Verilated::commandArgs(argc, argv);
  ddr = (uint8_t *)mmap(nullptr, DDR_BYTES, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  load(argv[1], 0x2f000000ull);
  load(argv[2], 0x3e000000ull);
  load(argv[3], 0x3f000000ull);
  std::vector<uint32_t> prompt;
  for (char *p = argv[4]; *p;) {
    prompt.push_back((uint32_t)strtoul(p, &p, 10));
    if (*p == ',') ++p;
  }
  uint32_t generate = (uint32_t)strtoul(argv[5], nullptr, 10);

  top = new Vsolo_core;
  top->clk_i = 0;
  top->fp_clk_i = 0;
  top->rx = 1;
  top->calib_done_i = 1;
  top->rst_ni = 0;
  top->fp_rst_ni = 0;
  axi_inputs();
  run(20);
  top->rst_ni = 1;
  top->fp_rst_ni = 1;
  run(20);
  rx_active = false;
  last_tx = top->tx;
  uart_in.clear();

  uart_out.push_back('P');
  uint8_t ack = get_byte(20000000ull);
  if (ack != 0x06) {
    fprintf(stderr, "no ping ACK: got 0x%02x at cycle %llu\n", ack,
            (unsigned long long)cycle);
    return 1;
  }
  char id[5] = {0};
  for (int i = 0; i < 4; ++i) id[i] = (char)get_byte();
  fprintf(stderr, "monitor: %s calib=%u at cycle %llu\n", id, get_byte(),
          (unsigned long long)cycle);

  uart_out.push_back('J');
  put_u32(0x1000);
  if (get_byte() != 0x06) { fprintf(stderr, "no jump ACK\n"); return 1; }

  uart_out.push_back('G');
  put_u32((uint32_t)prompt.size());
  put_u32(generate);
  for (uint32_t t : prompt) put_u32(t);
  uint64_t start = cycle;
  for (;;) {
    uint8_t kind = get_byte(4000000000ull);
    if (kind == 'E') break;
    if (kind == 'X') {
      uint32_t stage = get_u32(), status = get_u32();
      fprintf(stderr, "accelerator error at stage %u status 0x%08x\n", stage,
              status);
      return 1;
    }
    if (kind != 'T') { fprintf(stderr, "unexpected byte 0x%02x\n", kind); return 1; }
    uint32_t token = get_u32();
    uint32_t profile[6];
    for (auto &p : profile) p = get_u32();
    printf("token %u  embed %u norm0 %u layers %u norm_out %u argmax %u total %u "
           "(%.2f ms @200MHz)\n",
           token, profile[0], profile[1], profile[2], profile[3], profile[4],
           profile[5], profile[5] / 200000.0);
    fflush(stdout);
  }
  fprintf(stderr, "done in %llu cycles\n", (unsigned long long)(cycle - start));
  delete top;
  return 0;
}
