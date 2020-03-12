#!/usr/bin/env julia
contenders = chmod.(["baseline/bin.jl", "weakline/bin.py"], 0o777)

function benchmark(sourcefile, stringsfile)
    if !isfile(sourcefile)
        println("$sourcefile doesn't exist")
        exit(1)
    end

    if !isfile(stringsfile)
        println("$stringsfile doesn't exist")
        exit(1)
    end

    println("working on $sourcefile with $stringsfile")

    for (idx, contender) in enumerate(contenders)
        println("running $(split(contender, '/')[1])...")
        @time run(`./$contender $sourcefile $stringsfile output-$idx.txt`)
    end

    outputs = readlines.("output-$idx.txt" for idx in 1:length(contenders))
    rm.("output-$idx.txt" for idx in 1:length(contenders))

    for idx in 1:length(contenders)-1
        outputs[idx] == outputs[idx+1] && continue

        println("there is a disagreement in results")
        println(outputs[idx])
        println(outputs[idx+1])
        exit(1)
    end

    println("all the contenders agree")
end

benchmark("data/thaliana.fna", "data/2000markers.csv")
benchmark("data/thaliana.fna", "data/800000markers.csv")
