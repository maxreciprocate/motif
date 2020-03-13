#!/usr/bin/env julia
contenders = chmod.(["baseline/bin.jl", "weakline/bin.py"], 0o777)

function benchmark(sourcesfilename, stringsfilename)
    if !isfile(sourcesfilename)
        println("$sourcesfilename doesn't exist")
        exit(1)
    end

    if !isfile(stringsfilename)
        println("$stringsfilename doesn't exist")
        exit(1)
    end

    for (idx, contender) in enumerate(contenders)
        println("running $(split(contender, '/')[1])...")
        @time run(`./$contender $sourcesfilename $stringsfilename output-$idx.txt`)
    end

    output1 = read("output-1.txt")
    rm("output-1.txt")

    for idx in 2:length(contenders)
        output2 = read("output-$idx.txt")
        rm("output-$idx.txt")

        if output1 == output2
            output2 = output1
            continue
        end

        println("there is a disagreement in results of $(contenders[idx-1]) and $(contenders[idx])")
        exit(1)
    end

    println("all the contenders agree")
end

benchmark("data/1genomes.txt",  "data/2000markers.csv")
benchmark("data/20genomes.txt", "data/800000markers.csv")
