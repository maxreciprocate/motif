#!/usr/bin/env julia
import Base.match
using DataStructures: Queue, enqueue!
using Base.Threads: @threads

mutable struct Vertex
    children::Dict{Char, Vertex}
    key::Union{UInt32, Nothing}
    suffix::Union{Vertex, Nothing}
    isroot::Bool
    Z::UInt32

    Vertex() = new(Dict{Char, Vertex}(), nothing, nothing, false, 0)
end
# ■

function add!(vx::Vertex, string, key::UInt32)
    for char in string
        if !haskey(vx.children, char)
            vx.children[char] = Vertex()
        end

        vx = vx.children[char]
    end

    vx.key = key
end

function build!(root::Vertex)
    queue = Queue{Vertex}()
    enqueue!(queue, root)

    for vx in queue, (char, child) in vx.children
        child.suffix = if vx.isroot
            vx
        else
            search(vx.suffix, char)
        end

        enqueue!(queue, child)
    end
end

function search(vx::Vertex, char::Char)
    if haskey(vx.children, char)
        vx.children[char]
    elseif vx.isroot
        vx
    else
        search(vx.suffix, char)
    end
end

# ■

const T = Dict{Char, UInt8}('A' => 1, 'C' => 2, 'G' => 3, 'T' => 4)

function build_things(strfname::String)
    build1start = time()
    fsm = Vertex()
    fsm.suffix = fsm
    fsm.isroot = true

    sumlengths = 0

    strings = readlines(strfname)
    strings = map(line -> split(line, ',')[2], strings)

    for (idx, string) in enumerate(strings)
        add!(fsm, string, UInt32(idx))
        sumlengths += length(string)
    end

    build!(fsm)
    build1time = time() - build1start
    println("finished dict building in $build1time")

    build2start = time()

    queue = Queue{Vertex}()
    enqueue!(queue, fsm)
    P = UInt32(0)
    fsm.Z = P
    X = zeros(UInt32, (4, sumlengths+1))

    ENDS = Dict{UInt32, Vector{UInt32}}()

    for vx in queue
        thisP = vx.Z
        for C in ['A', 'C', 'G', 'T']
            if haskey(vx.children, C)
                child = vx.children[C]
                P += 1
                # new alloc
                X[4*thisP + T[C]] = P
                # save position
                child.Z = P

                evx = child
                while true
                    if evx.key != nothing
                        haskey(ENDS, P) || (ENDS[P] = Vector{UInt32}())
                        push!(ENDS[P], evx.key)
                    end

                    evx.isroot && break
                    evx = evx.suffix
                end

                enqueue!(queue, vx.children[C])
            else
                # inline here
                X[4*thisP + T[C]] = search(vx.suffix, C).Z
            end

        end
    end

    build2time = time() - build2start
    println("finished table building in $build2time")


    X, ENDS, length(strings)
end

function match(ssfname::String, strfname::String, outfname::String)
    X, ENDS, N = build_things(strfname)
    outputf = open(outfname, "w")

    @threads for sfname in readlines(ssfname)
        source = readline("data/$sfname")
        P = UInt32(0)
        output = zeros(UInt8, N)

        for C in source
            if C == 'N'
                P = UInt32(0)
                continue
            end

            P = X[4*P + T[C]]

            if haskey(ENDS, P)
                for key in ENDS[P]
                    output[key] = 0x01
                end
            end
        end

        println("done with $sfname")

        formattedoutput = split(sfname, '/')[end] * " " * String(output .+ UInt8('0')) * '\n'
        write(outputf, formattedoutput)
    end

    close(outputf)
end

if length(ARGS) != 3
    println("# usage: julia bin.jl <sourcefile> <stringsfile> <outputfile>")
    exit(1)
end

println("We have this many threads = $(Threads.nthreads())")
match(ARGS...)
