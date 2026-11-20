# run_hls.tcl - Flujo Tcl equivalente al Makefile individual, para equipos
# que solo tienen Vitis HLS (sin v++ ni vitis-run). Lee hls.cfg del kernel.
# Uso (desde la carpeta del kernel):
#   vitis_hls -f ../run_hls.tcl -tclargs <csim|synth|cosim|export>
# Reportes: build/sol1/syn/report/<top>_csynth.rpt y build/sol1/sim/report/<top>_cosim.rpt

set step [lindex $argv end]

set part ""; set clock ""; set top ""; set src {}; set tb {}
set fp [open hls.cfg r]
foreach line [split [read $fp] "\n"] {
    if {[regexp {^\s*([A-Za-z_.]+)\s*=\s*(\S+)\s*$} $line -> key value]} {
        switch -- $key {
            part     { set part $value }
            clock    { set clock $value }
            syn.top  { set top $value }
            syn.file { lappend src $value }
            tb.file  { lappend tb $value }
        }
    }
}
close $fp
regsub {ns$} $clock "" clock
puts "run_hls: top=$top part=$part clock=${clock}ns step=$step"

if {$step eq "csim" || $step eq "synth"} {
    open_project -reset build
    set_top $top
    foreach f $src { add_files $f }
    foreach f $tb  { add_files -tb $f }
    open_solution -reset sol1 -flow_target vitis
    set_part $part
    create_clock -period $clock
} elseif {$step eq "cosim" || $step eq "export"} {
    open_project build
    open_solution sol1
} else {
    puts "Uso: vitis_hls -f ../run_hls.tcl -tclargs <csim|synth|cosim|export>"
    exit 1
}

switch -- $step {
    csim   { csim_design }
    synth  { csynth_design }
    cosim  { cosim_design }
    export { export_design -format xo -output [file normalize build/${top}.xo] }
}
exit
