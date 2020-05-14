#!/usr/bin/env julia
inittime = time()
using CuArrays
using CUDAnative
using Printf: @printf

function pptable(table::Vector{T}) where T
    @printf "%4s %3s %3s %3s %3s\n" "A" "C" "G" "T" "X"
    nrows = 50
    for idx = 0:min(length(table) ÷ 5-1, nrows)
        for ridx = 1:5
            @printf "%4d" table[5 * idx + ridx]
        end

        println()
    end
end

const T = Dict{Char, UInt8}(zip(['A', 'C', 'G', 'T'], 1:4))
const Lut = [get(T, Char(char), 0x0) for char in UInt('A'):UInt('T')]

function build(markersfn::String)
    markers = last.(split.(readlines(markersfn), ','))
    nchars = sum(length.(markers))
    nmarkers = length(markers)

    tablesize = ceil(Int, nchars - nmarkers * log(4, nmarkers / √4) + 4)
    table = zeros(UInt32, tablesize * 5)

    edge = 0x0
    for (id, marker) in enumerate(markers)
        vx = 0x0

        for char in marker
            idx = 5 * vx + T[char]

            if table[idx] == 0
                edge += 1
                table[idx] = edge
            end

            vx = table[idx]
        end

        table[5 * vx + 5] = id
    end

    table, nmarkers
end

function latch!(table_d, l, source_d, output_d, lutted_d)
    tidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    ℓ = l[1]

    # @cuprintf "stride: %d\n" stride
    # @cuprintf "tidx: %d\n" tidx

    while tidx <= ℓ
        char = source_d[tidx]
        # @cuprintf "%d, processing %c\n" tidx char

        if char == 0x4e
            tidx += stride
            continue
        end

        vx = table_d[lutted_d[char - 0x40]]
        idx = tidx

        while true
            wordsidx = table_d[5 * vx + 5]

            if wordsidx != 0
                output_d[wordsidx] = 0x1
            end

            idx += 1
            if idx > length(source_d) || vx == 0
                break
            end

            char = source_d[idx]
            char == 0x4e && break
            vx = table_d[5 * vx + lutted_d[char - 0x40]]
        end

        tidx += stride
    end

    return
end

function match(sourcesfn::String, markersfn::String, outputfn::String)
    @time table, nmarkers = build(markersfn)

    prefixdir = join(split(sourcesfn, '/')[1:end-1], '/')
    outputf = open(outputfn, "w")

    copytime = time()
    sourcefn = readline(sourcesfn)
    source_h = read(joinpath(prefixdir, sourcefn))
    source_d = CuArray(source_h)
    output_d = CuArray(zeros(UInt8, nmarkers))
    lutted_d = CuArray(Lut)
    table_d  = CuArray(table)
    ℓ = CuArray(UInt32[length(source_h)])

    @printf "(copy) %.2fs\n" time() - copytime

    @time @cuda threads=1024 blocks=16 latch!(table_d, ℓ, source_d, output_d, lutted_d)

    output = Array(output_d)

    remaketime = time()
    markers = last.(split.(readlines(markersfn), ','))
    checked = markers[findall(x -> x == 1, output)]

    for (id, marker) in enumerate(markers)
        if findfirst(isequal(marker), checked) != nothing
            output[id] = 0x31
        else
            output[id] = 0x30
        end
    end

    write(outputf, "$(last(split(sourcefn, '/'))) " * String(output) * '\n')
    @printf "(remake) %.2fs\n" time() - remaketime
    close(outputf)
end

@printf "(lost) %.2fs\n" time() - inittime
@time match(ARGS...)
