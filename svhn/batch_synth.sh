#/bin/sh

function synth {

    name=${1%.tar.gz}

    if [ -f rpt-$name.tgz ]; then
        exit
    fi
    if [ -f $name.crash.log ]; then
        exit
    fi


    tar -xzf $1 --one-top-level=$name.tmp

    loc=$(dirname "$(find $name.tmp -type d -name firmware)")
    
    mv $loc $name

    if [ ! $? -eq 0 ]; then
        echo "Error: failed to extract $1"
        exit
    fi

    find $name.tmp -type d -empty -delete

    if [ -d $name.tmp ]; then
        echo "Error: directory structure of $1 is not as expected"
        exit
    fi

    cd $name
    if [ ! $? -eq 0 ]; then
        echo "Error: failed to enter $name"
        exit
    fi

    perl -0777 -pi -e 's/(?<=array set opt \{)[^\}]+(?=\})/\n    reset      0\n    csim       0\n    synth      1\n    cosim      0\n    validation 0\n    export     1\n    vsynth     1\n    fifo_opt   0\n/' build_prj.tcl
    perl -pi -e "s/export_design -format/export_design -flow impl -rtl verilog -format/" build_prj.tcl
    perl -pi -e "s/#pragma HLS (?:PIPELINE|DATAFLOW)/#pragma HLS DATAFLOW" firmware/*.cpp

    vivado_hls -f build_prj.tcl > ./synth.log

    if [ ! $? -eq 0 ]; then
        echo "Error: synth failed"
        ln synth.log ../$name.crash.log
        exit
    fi

    cd ..

    find $name -not \( -name '*.rpt' -o -name '*.xml' -o -name 'synth.log' -o -name 'keras_model.h5' \) -type f -delete
    find $name -type d -empty -delete

    tar -czf rpt-$name.tgz $name
    if [ ! $? -eq 0 ]; then
        echo "Error: failed to create rpt-$name.taz"
        exit
    fi
    rm -rf $name

    echo "Success: $name"
}


for arg in "$@"
do
    # Check if argument starts with -j
    if [[ $arg == -j* ]]; then
        # Extract number after -j and assign to thread
        thread=${arg#-j}
    else
        # Append argument to args string with newline separator
        if [ -f "$arg" ]; then
            fs+="$arg\n"
        else
            echo "Error: $arg does not exist"
            exit
        fi
    fi
done
fs=${fs%\\n}

export -f synth

# echo number of files recorded
echo "Files: $(echo -e $fs | wc -l)"
echo "Threads: $thread"
echo -e $fs | parallel --progress -j $thread synth {}
