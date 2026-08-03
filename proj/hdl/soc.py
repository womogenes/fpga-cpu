#!/usr/bin/env python3

from migen import Cat, ClockDomain, ClockDomainsRenamer, If, Mux, Signal
from migen.genlib.cdc import MultiReg

from litex.gen import LiteXModule
from litex.build.io import DifferentialInput
from litex.soc.cores.clock import S7IDELAYCTRL, S7MMCM
from litex.soc.cores.video import (
    VideoS7HDMIPHY,
    VideoTimingGenerator,
    video_data_layout,
    video_framebuffer_size,
    video_timing_layout,
)
from litex.soc.integration.builder import Builder
from litex.soc.integration.soc import SoCCore
from litex.soc.interconnect import stream

from litex_boards.platforms import digilent_genesys2

from litedram.modules import MT41J256M16
from litedram.phy import s7ddrphy


SYS_CLK_FREQ = 100e6
VIDEO_TIMINGS = "1280x720@60Hz"
FRAMEBUFFER_BASE = 0x7FE00000


class TimingSafeVideoFrameBuffer(LiteXModule):
    """LiteDRAM framebuffer reader that never stalls the video timing."""

    def __init__(
        self,
        port,
        hres=1280,
        vres=720,
        base=FRAMEBUFFER_BASE,
        fifo_depth=64 * 1024,
        clock_domain="hdmi",
    ):
        from litedram.frontend.dma import LiteDRAMDMAReader

        self.vtg_sink = vtg_sink = stream.Endpoint(video_timing_layout)
        self.source = source = stream.Endpoint(video_data_layout)
        self.underflow = Signal()
        self.depth = 16

        dma_fifo_depth = fifo_depth // (port.data_width // 8)
        self.dma = LiteDRAMDMAReader(
            port,
            fifo_depth=dma_fifo_depth,
            fifo_buffered=True,
        )
        self.dma.add_csr(
            default_base=base,
            default_length=hres * vres * 2,
            default_enable=0,
            default_loop=1,
        )

        # The system clock is faster than the 74.25 MHz pixel clock, so split
        # each native DRAM word into RGB565 pixels before crossing domains.
        self.converter = stream.Converter(port.data_width, self.depth)
        self.cdc = stream.ClockDomainCrossing(
            [("data", self.depth)],
            cd_from="sys",
            cd_to=clock_domain,
            depth=1024,
            buffered=True,
        )
        self.comb += [
            self.dma.source.connect(self.converter.sink),
            self.converter.source.connect(self.cdc.sink),
        ]
        pixel = self.cdc.source

        # Let the DMA and cross-clock FIFO prefill, then begin consuming at a
        # frame boundary. The stock four-entry CDC FIFO exposes normal DRAM
        # burst gaps directly as visible underflow at 720p.
        dma_enable = Signal()
        running = Signal()
        self.specials += MultiReg(self.dma.enable, dma_enable, clock_domain)
        video_sync = getattr(self.sync, clock_domain)
        video_sync += If(
            ~dma_enable,
            running.eq(0),
        ).Elif(
            ~running & vtg_sink.valid & vtg_sink.first & pixel.valid,
            running.eq(1),
        )
        render = Signal()
        self.comb += render.eq(
            running | (vtg_sink.valid & vtg_sink.first & pixel.valid)
        )

        # HDMI timing must remain continuous. Consume one pixel only during
        # active video, substituting black if the DMA stream ever underflows.
        self.comb += [
            vtg_sink.connect(
                source,
                keep={"valid", "ready", "de", "hsync", "vsync"},
            ),
            pixel.ready.eq(
                render & vtg_sink.valid & vtg_sink.de & source.ready
            ),
            self.underflow.eq(
                render & vtg_sink.valid & vtg_sink.de & ~pixel.valid
            ),
            If(
                render & pixel.valid,
                source.r.eq(Cat(Signal(3), pixel.data[11:16])),
                source.g.eq(Cat(Signal(2), pixel.data[5:11])),
                source.b.eq(Cat(Signal(3), pixel.data[0:5])),
            ).Else(
                source.r.eq(0),
                source.g.eq(0),
                source.b.eq(0),
            ),
        ]


class CRG(LiteXModule):
    def __init__(self, platform, sys_clk_freq):
        self.rst = Signal()

        self.cd_sys = ClockDomain()
        self.cd_sys4x = ClockDomain()
        self.cd_idelay = ClockDomain()
        self.cd_hdmi = ClockDomain()
        self.cd_hdmi5x = ClockDomain()

        clk200_pads = platform.request("clk200")
        cpu_reset_n = platform.request("cpu_reset_n")
        btn0 = platform.request("user_btn_c")
        clk200 = Signal()
        reset = Signal()
        self.specials += DifferentialInput(clk200_pads.p, clk200_pads.n, clk200)
        self.comb += reset.eq(~cpu_reset_n | btn0 | self.rst)

        self.sys_mmcm = sys_mmcm = S7MMCM(speedgrade=-2)
        self.comb += sys_mmcm.reset.eq(reset)
        sys_mmcm.register_clkin(clk200, 200e6)
        sys_mmcm.create_clkout(self.cd_sys, sys_clk_freq)
        sys_mmcm.create_clkout(self.cd_sys4x, 4 * sys_clk_freq)
        sys_mmcm.create_clkout(self.cd_idelay, 200e6)
        platform.add_false_path_constraints(self.cd_sys.clk, sys_mmcm.clkin)

        self.idelayctrl = S7IDELAYCTRL(self.cd_idelay)

        # CEA-720p60 uses a 74.25MHz pixel clock and a 5x DDR serializer clock.
        self.video_mmcm = video_mmcm = S7MMCM(speedgrade=-2)
        self.comb += video_mmcm.reset.eq(reset)
        video_mmcm.register_clkin(clk200, 200e6)
        video_mmcm.create_clkout(self.cd_hdmi, 74.25e6, margin=0)
        video_mmcm.create_clkout(self.cd_hdmi5x, 5 * 74.25e6, margin=0)

        # Crossings between the unrelated MMCMs use synchronizers and async FIFOs.
        platform.add_false_path_constraints(self.cd_sys.clk, self.cd_hdmi.clk)


class Genesys2SoC(SoCCore):
    def add_timing_safe_video_framebuffer(self, phy):
        framebuffer_size = video_framebuffer_size(1280, 720, "rgb565")
        framebuffer_region = self._get_video_framebuffer_region(
            name="video_framebuffer",
            size=framebuffer_size,
            region_name="main_ram",
            base=FRAMEBUFFER_BASE,
        )
        dma_port, _ = self._get_video_framebuffer_dma_port(
            region_name="main_ram",
        )

        vtg = ClockDomainsRenamer("hdmi")(
            VideoTimingGenerator(default_video_timings=VIDEO_TIMINGS)
        )
        vfb = TimingSafeVideoFrameBuffer(
            dma_port,
            base=framebuffer_region.origin,
            clock_domain="hdmi",
        )

        self.bus.add_region("video_framebuffer", framebuffer_region)
        self.add_module(name="video_framebuffer_vtg", module=vtg)
        self.add_module(name="video_framebuffer", module=vfb)
        self.comb += [
            vtg.source.connect(vfb.vtg_sink),
            vfb.source.connect(phy),
        ]
        self.add_constant("VIDEO_FRAMEBUFFER_BASE", framebuffer_region.origin)
        self.add_constant("VIDEO_FRAMEBUFFER_HRES", 1280)
        self.add_constant("VIDEO_FRAMEBUFFER_VRES", 720)
        self.add_constant("VIDEO_FRAMEBUFFER_DEPTH", vfb.depth)

    def __init__(self, sys_clk_freq=SYS_CLK_FREQ, toolchain="vivado", **kwargs):
        platform = digilent_genesys2.Platform(toolchain=toolchain)

        self.crg = CRG(platform, sys_clk_freq)

        SoCCore.__init__(
            self,
            platform,
            sys_clk_freq,
            ident="Genesys 2 LiteX CPU with 720p framebuffer",
            **kwargs,
        )

        if not self.integrated_main_ram_size:
            self.ddrphy = s7ddrphy.K7DDRPHY(
                platform.request("ddram"),
                memtype="DDR3",
                nphases=4,
                sys_clk_freq=sys_clk_freq,
            )
            self.add_sdram(
                "sdram",
                phy=self.ddrphy,
                module=MT41J256M16(sys_clk_freq, "1:4"),
                l2_cache_size=kwargs.get("l2_size", 8192),
            )

        self.videophy = VideoS7HDMIPHY(
            platform.request("hdmi_out"),
            clock_domain="hdmi",
        )

        # Keep an independent color-bar path in the full SoC so HDMI and the
        # DDR framebuffer can be tested with the same bitstream. Physical SW0
        # is LiteX resource user_sw[7]; high selects the framebuffer and low
        # selects color bars.
        framebuffer_video = stream.Endpoint(video_data_layout)
        colorbars_video = stream.Endpoint(video_data_layout)
        self.add_timing_safe_video_framebuffer(framebuffer_video)
        self.add_video_colorbars(
            phy=colorbars_video,
            timings=VIDEO_TIMINGS,
            clock_domain="hdmi",
        )

        framebuffer_select = Signal()
        self.specials += MultiReg(
            platform.request("user_sw", 7),
            framebuffer_select,
            "hdmi",
        )
        self.comb += [
            framebuffer_video.ready.eq(framebuffer_select & self.videophy.sink.ready),
            colorbars_video.ready.eq(~framebuffer_select & self.videophy.sink.ready),
            self.videophy.sink.valid.eq(Mux(
                framebuffer_select,
                framebuffer_video.valid,
                colorbars_video.valid,
            )),
            self.videophy.sink.hsync.eq(Mux(
                framebuffer_select,
                framebuffer_video.hsync,
                colorbars_video.hsync,
            )),
            self.videophy.sink.vsync.eq(Mux(
                framebuffer_select,
                framebuffer_video.vsync,
                colorbars_video.vsync,
            )),
            self.videophy.sink.de.eq(Mux(
                framebuffer_select,
                framebuffer_video.de,
                colorbars_video.de,
            )),
            self.videophy.sink.r.eq(Mux(
                framebuffer_select,
                framebuffer_video.r,
                colorbars_video.r,
            )),
            self.videophy.sink.g.eq(Mux(
                framebuffer_select,
                framebuffer_video.g,
                colorbars_video.g,
            )),
            self.videophy.sink.b.eq(Mux(
                framebuffer_select,
                framebuffer_video.b,
                colorbars_video.b,
            )),
        ]

        # Physical LED labels run opposite LiteX's resource indices on this
        # board. LED0/1 show system/video PLL lock. LED2 toggles each frame;
        # LED3/4/5 show framebuffer DE/valid/underflow, and LED7 shows SW0.
        user_leds = [platform.request("user_led", i) for i in range(8)]
        framebuffer_vsync_d = Signal()
        framebuffer_heartbeat = Signal()
        self.sync.hdmi += [
            framebuffer_vsync_d.eq(framebuffer_video.vsync),
            If(framebuffer_video.vsync & ~framebuffer_vsync_d,
                framebuffer_heartbeat.eq(~framebuffer_heartbeat),
            ),
        ]
        self.comb += [
            user_leds[0].eq(framebuffer_select),
            user_leds[1].eq(self.video_framebuffer.underflow),
            user_leds[2].eq(framebuffer_video.valid),
            user_leds[3].eq(framebuffer_video.de),
            user_leds[4].eq(framebuffer_heartbeat),
            user_leds[5].eq(0),
            user_leds[6].eq(self.crg.video_mmcm.locked),
            user_leds[7].eq(self.crg.sys_mmcm.locked),
        ]


def main():
    from litex.build.parser import LiteXArgumentParser

    parser = LiteXArgumentParser(
        platform=digilent_genesys2.Platform,
        description="Minimal LiteX CPU and HDMI framebuffer for the Genesys 2.",
    )
    parser.add_target_argument(
        "--sys-clk-freq",
        default=SYS_CLK_FREQ,
        type=float,
        help="System clock frequency.",
    )
    args = parser.parse_args()

    soc = Genesys2SoC(
        sys_clk_freq=args.sys_clk_freq,
        toolchain=args.toolchain,
        **parser.soc_argdict,
    )
    builder = Builder(soc, **parser.builder_argdict)
    if args.build:
        builder.build(**parser.toolchain_argdict)


if __name__ == "__main__":
    main()
