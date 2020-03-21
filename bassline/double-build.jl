#!/usr/bin/env julia
import Base.match
using DataStructures: Queue, enqueue!
using Base.Threads: @threads, nthreads

const T = Dict{Char, UInt8}(zip(['A', 'C', 'G', 'T'], 1:4))
const Lut = [get(T, Char(char), 0x0) for char in UInt('A'):UInt('T')]
const Root = 0x0

mutable struct Vertex
    isroot::Bool
    position::UInt32
    keys::Vector{UInt32}
    suffix::Union{Nothing, Vertex}
    children::Vector{Union{Nothing, Vertex}}

    Vertex() = new(false, 0, [], nothing, Vector{Union{Nothing, Vertex}}(nothing, 4))
end

function add!(vx::Vertex, string::SubString{String}, key::Int)
    for char in string
        if vx.children[T[char]] == nothing
            vx.children[T[char]] = Vertex()
        end

        vx = vx.children[T[char]]
    end

    push!(vx.keys, key)
end

function search(vx::Vertex, char::UInt8)
    if vx.children[char] != nothing
        vx.children[char]
    elseif vx.isroot
        vx
    else
        search(vx.suffix, char)
    end
end

function build!(root::Vertex)
    queue = Queue{Vertex}()
    enqueue!(queue, root)

    for vx in queue, (char, child) in enumerate(vx.children)
        child == nothing && continue

        child.suffix = if vx.isroot
            vx
        else
            search(vx.suffix, UInt8(char))
        end

        enqueue!(queue, child)
    end
end

function create(markersfn::String)
    fsm = Vertex()
    fsm.isroot = true

    markers = last.(split.(readlines(markersfn), ','))
    sumlengths = 0x0

    for (idx, marker) in enumerate(markers)
        add!(fsm, marker, idx)
        sumlengths += length(marker)
    end

    build!(fsm)

    queue = Queue{Vertex}()
    enqueue!(queue, fsm)
    fsm.position = Root

    table = zeros(UInt32, 4 * (sumlengths+1))
    words = Dict{UInt32, Vector{UInt32}}()

    edge = Root
    for vx in queue, char in 0x1:0x4
        if vx.children[char] != nothing
            child = vx.children[char]
            enqueue!(queue, child)

            edge += 1
            table[4 * vx.position + char] = edge
            child.position = edge

            evx = child
            while true
                if !isempty(evx.keys)
                    haskey(words, edge) || (words[edge] = Vector{UInt32}())
                    for word in evx.keys
                        push!(words[edge], word)
                    end
                end

                evx.isroot && break
                evx = evx.suffix
            end

        elseif vx.suffix != nothing
            table[4 * vx.position + char] = search(vx.suffix, char).position
        end
    end

    table, words, length(markers)
end

function match(sourcesfn::String, markersfn::String, outputfn::String)
    buildstart = time()
    table, words, nmarkers = create(markersfn)
    println("building took $(time() - buildstart)")

    prefixdir = join(split(sourcesfn, '/')[1:end-1], '/')
    outputf = open(outputfn, "w")

    @threads for sourcefn in readlines(sourcesfn)
        output = zeros(UInt8, nmarkers)

        vx = Root
        for char in read(joinpath(prefixdir, sourcefn))
            if char == 0x4e
                vx = Root
                continue
            end

            @inbounds vx = table[4 * vx + Lut[char - 0x40]]

            if haskey(words, vx)
                for word in words[vx]
                    @inbounds output[word] = 0x1
                end
            end
        end

        write(outputf, "$(last(split(sourcefn, '/'))) " * String(output .+ 0x30) * '\n')
    end

    close(outputf)
end

if length(ARGS) != 3
    println("# usage: julia $PROGRAM_FILE <sourcefile> <stringsfile> <outputfile>")
    exit(1)
end

match(ARGS...)
