#!/usr/bin/env julia
contenders = [
    #("bassline", "bassline/double-build.jl"),
    ("groove", "motif/build/groove"),
    ("jam", "jam/jam"),
    ("jamlib", "python_interface/run.py")
]

chmod.(last.(contenders), 0o777)

function benchmark(sources::Vector{String})
    if !all(isfile.(sources))
        println("bench data is not present")
        exit(1)
    end

    timings = zeros(length(contenders))

    for (idx, (name, exe)) in enumerate(contenders)
        start = time()
        run(`./$exe $(sources[1]) $(sources[2]) output-$idx.txt`)
        timings[idx] = round(time() - start; digits=2)

        println("$name = $(timings[idx])")
    end

    output1 = Dict{String, String}()
    for line in readlines("output-1.txt")
        genome, string = split(line, ' ')
        output1[genome] = string
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

    println("all contenders agree")

    timings
end

benchmarks = [
    ["data/1genomes.txt", "data/8000markers.csv"],
    #["data/10genomes.txt", "data/80000markers.csv"],
    #["data/100genomes.txt", "data/800000markers.csv"],
    #["data/1000genomes.txt", "data/3000000markers.csv"],
    ["data/1genomes.txt", "data/markers.csv"],
]

if length(ARGS) > 0
    try
        nbenchs = round(Int, log10(parse(Int, ARGS[1]))) + 1

        if nbenchs <= size(benchmarks)[1]
            global benchmarks = benchmarks[1:nbenchs]
        end
    catch
        println("usage: julia download.jl <number of benches>")
        exit(1)
    end
end

ENV["JULIA_NUM_THREADS"] = ENV["PYPY_NUM_THREADS"] = 4
ENV["SAVE_RESULT_TO_FILE"] = 1

results = benchmark.(benchmarks)

for (idx, timings) in enumerate(eachrow(hcat(results...)))
    println("$(first(contenders[idx])) = $timings")
end
