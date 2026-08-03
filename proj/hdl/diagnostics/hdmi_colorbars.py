#!/usr/bin/env python3

from migen import ClockDomain, Signal

from litex.build.io import DifferentialInput
from litex.gen import LiteXModule
from litex.soc.cores.clock import S7MMCM
from litex.soc.cores.video import VideoS7HDMIPHY
from litex.soc.integration.builder import Builder
from litex.soc.integration.soc import SoCMini
from litex_boards.platforms import digilent_genesys2


class CRG(LiteXModule):
    def __init__(self, platform):
        self.cd_sys = ClockDomain()
        self.cd_hdmi = ClockDomain()
        self.cd_hdmi5x = ClockDomain()

        clk200_pads = platform.request("clk200")
        cpu_reset_n = platform.request("cpu_reset_n")
        btn0 = platform.request("user_btn_c")
        clk200 = Signal()
        reset = Signal()

        self.specials += DifferentialInput(clk200_pads.p, clk200_pads.n, clk200)
        self.comb += reset.eq(~cpu_reset_n | btn0)

        self.sys_pll = sys_pll = S7MMCM(speedgrade=-2)
        self.comb += sys_pll.reset.eq(reset)
        sys_pll.register_clkin(clk200, 200e6)
        sys_pll.create_clkout(self.cd_sys, 100e6)

        self.video_pll = video_pll = S7MMCM(speedgrade=-2)
        self.comb += video_pll.reset.eq(reset)
        video_pll.register_clkin(clk200, 200e6)
        video_pll.create_clkout(self.cd_hdmi, 74.25e6)
        video_pll.create_clkout(self.cd_hdmi5x, 5 * 74.25e6)

        platform.add_false_path_constraints(self.cd_sys.clk, self.cd_hdmi.clk)


class HDMIColorbarsSoC(SoCMini):
    def __init__(self):
        platform = digilent_genesys2.Platform(toolchain="vivado")
        self.crg = CRG(platform)

        SoCMini.__init__(
            self,
            platform,
            100e6,
            ident="Genesys 2 HDMI color-bars diagnostic",
        )

        self.videophy = VideoS7HDMIPHY(
            platform.request("hdmi_out"),
            clock_domain="hdmi",
        )
        self.add_video_colorbars(
            phy=self.videophy,
            timings="1280x720@60Hz",
            clock_domain="hdmi",
        )

        self.comb += [
            platform.request("user_led", 7).eq(self.crg.sys_pll.locked),
            platform.request("user_led", 6).eq(self.crg.video_pll.locked),
        ]


def main():
    soc = HDMIColorbarsSoC()
    builder = Builder(soc, output_dir="build/genesys2_hdmi_diag")
    builder.build(build_name="genesys2_hdmi_diag")


if __name__ == "__main__":
    main()
