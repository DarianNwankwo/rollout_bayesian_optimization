#!/bin/bash
for i in 0 1 2 3 4; do
    if [ $i -eq 0 ]; then
        arg1=0
        arg2=250
    elif [ $i -eq 1 ]; then
        arg1=1
        arg2=350
    elif [ $i -eq 2 ]; then
        arg1=2
        arg2=500
    elif [ $i -eq 3 ]; then
        arg1=3
        arg2=500
    elif [ $i -eq 4 ]; then
        arg1=4
        arg2=500
    fi
    julia nonmyopic_acquisition_plots.jl $arg1 $arg2 &
    echo "$arg1,$arg2 contrived experiment started"
done
