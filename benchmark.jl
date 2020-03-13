#!/usr/bin/env julia
contenders = chmod.(["bassline/bin.jl", "sub-bass/bin.py"], 0o777)

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

    output1 = Dict{String, String}()
    for line in readlines("output-1.txt")
        genome, presence = split(line, ' ')
        output1[genome] = presence
    end

    for idx in 2:length(contenders)
        output2 = Dict{String, String}()
        for line in readlines("output-$idx.txt")
            genome, string = split(line, ' ')
            output2[genome] = string
        end

        for (genome, string) in output1
            if output1[genome] != output2[genome]
                println("there is a disagreement in results of $(contenders[idx-1]) and $(contenders[idx])")
                exit(1)
            end
        end

        output1 = output2
    end

    println("all the contenders agree")
end

import REPL
using REPL.TerminalMenus

benches = ["data/1genomes.txt data/2000markers.csv", "data/20genomes.txt data/800000markers.csv"]

benchmark(split(benches[request(RadioMenu(benches))], ' ')...)
