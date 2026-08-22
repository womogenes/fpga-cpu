set project "hardware/corev_apu/fpga/ariane.xpr"

open_project $project
reset_run impl_1
set run [get_runs impl_1]
set_property strategy Performance_ExplorePostRoutePhysOpt $run
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE ExtraTimingOpt $run
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore $run
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE Explore $run
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore $run
launch_runs impl_1 -to_step write_bitstream -jobs 1
wait_on_run impl_1
close_project
