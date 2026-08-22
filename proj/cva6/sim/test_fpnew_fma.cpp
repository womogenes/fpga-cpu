#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <random>

#include "Vfpnew_fma_wrapper.h"
#include "verilated.h"

struct Expected {
  uint8_t tag;
  uint32_t bits;
};

static uint32_t bits(float value) { return std::bit_cast<uint32_t>(value); }

static void tick(Vfpnew_fma_wrapper &dut) {
  dut.clk_i = 0;
  dut.eval();
  dut.clk_i = 1;
  dut.eval();
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  Vfpnew_fma_wrapper dut;
  dut.rst_ni = 0;
  dut.in_valid_i = 0;
  dut.out_ready_i = 1;
  for (int cycle = 0; cycle < 4; ++cycle) tick(dut);
  dut.rst_ni = 1;

  std::mt19937 generator(0xc0a6f00du);
  std::uniform_real_distribution<float> values(-8.0f, 8.0f);
  std::deque<Expected> expected;
  unsigned submitted = 0;
  unsigned received = 0;
  constexpr unsigned kOperations = 20000;

  while (received < kOperations) {
    dut.out_ready_i = (generator() & 7u) != 0u;
    dut.in_valid_i = submitted < kOperations;
    float a = values(generator);
    float b = values(generator);
    float c = values(generator);
    unsigned operation = submitted % 3u;
    dut.operand_a_i = bits(a);
    dut.operand_b_i = bits(b);
    dut.operand_c_i = bits(c);
    dut.operation_i = operation;
    dut.tag_i = submitted & 0xffu;
    dut.clk_i = 0;
    dut.eval();
    bool accept_input = dut.in_valid_i && dut.in_ready_o;
    bool accept_output = dut.out_valid_o && dut.out_ready_i;
    if (accept_output) {
      if (expected.empty() || dut.tag_o != expected.front().tag ||
          dut.result_o != expected.front().bits) {
        std::fprintf(stderr,
                     "FAIL at result %u: tag=%02x bits=%08x expected=%02x/%08x\n",
                     received, dut.tag_o, dut.result_o,
                     expected.empty() ? 0xffu : expected.front().tag,
                     expected.empty() ? 0u : expected.front().bits);
        return 1;
      }
      expected.pop_front();
      ++received;
    }
    if (accept_input) {
      float result = operation == 0u ? b + c
                     : operation == 1u ? a * b
                                       : std::fma(a, b, c);
      expected.push_back({static_cast<uint8_t>(submitted), bits(result)});
      ++submitted;
    }
    dut.clk_i = 1;
    dut.eval();
  }
  std::printf("PASS: %u back-to-back FP32 ADD/MUL/FMA operations\n", received);
  return 0;
}
