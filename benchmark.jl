#!/usr/bin/env julia
using Plots
contenders = chmod.(["sub-bass/bin.py", "bassline/bin.jl"], 0o777)

function benchmark(sources::Vector{String})
    if !all(isfile.(sources))
        println("bench data is not present")
        exit(1)
    end

    timings = zeros(length(contenders))

    for (idx, contender) in enumerate(contenders)
        println("running $(split(contender, '/')[1])...")
        start = time()
        run(`./$contender $(sources[1]) $(sources[2]) output-$idx.txt`)
        timings[idx] = time() - start
        println(timings[idx])
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

    timings
end

ENV["JULIA_NUM_THREADS"] = ENV["PYPY_NUM_THREADS"] = 4

benchmarks = [
    ["data/1genomes.txt", "data/8000markers.csv"],
    ["data/10genomes.txt", "data/80000markers.csv"],
    ["data/100genomes.txt", "data/800000markers.csv"],
    ["data/1000genomes.txt", "data/8000000markers.csv"],
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

results = benchmark.(benchmarks)

plot(dpi=200)

for (idx, timings) in enumerate(eachrow(hcat(results...)))
    plot!(timings, label=first(split(contenders[idx], '/')), legend=:topleft)
    println(timings)
end

nbenchs = size(results)[1]
xticks!([1:nbenchs;], ["$(10^i),$(10^i * 8000)" for i in 0:nbenchs-1])
ylabel!("seconds")
xlabel!("genomes & markers")
savefig("timings.png")
