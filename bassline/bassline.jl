#!/usr/bin/env julia
import Base.match
using DataStructures
using Test
using Base.Threads

mutable struct Vertex
    children::Dict{Char, Vertex}
    key::Union{Int, Nothing}
    suffix::Union{Vertex, Nothing}
    isroot::Bool

    Vertex() = new(Dict{Char, Vertex}(), nothing, nothing, false)
end

function add!(vx::Vertex, string, key::Int)
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

function match(source::String, strings)::String
    fsm = Vertex()
    fsm.isroot = true

    for (idx, string) in enumerate(strings)
        add!(fsm, string, idx)
    end

    build!(fsm)
    output = zeros(UInt8, length(strings))

    for char in source
        fsm = search(fsm, char)
        vx = fsm

        while true
            if vx.key != nothing
                output[vx.key] = 1
            end

            vx.isroot && break
            vx = vx.suffix
        end
    end

    output .+ UInt8('0') |> String
end

function match(sourcesfilename::String, stringsfilename::String, outputfilename::String)
    println("building")
    fsm = Vertex()
    fsm.isroot = true

    strings = readlines(stringsfilename)
    strings = map(line -> split(line, ',')[2], strings)

    for (idx, string) in enumerate(strings)
        add!(fsm, string, idx)
    end

    build!(fsm)
    println("finished")
    outputfile = open(outputfilename, "w")

    @threads for sourcefilename in readlines(sourcesfilename)
        output = zeros(UInt8, length(strings))
        source = readline("data/" * sourcefilename)

        vx = fsm
        for char in source
            vx = search(vx, char)
            evx = vx

            while true
                if evx.key != nothing
                    output[evx.key] = 1
                end

                evx.isroot && break
                evx = evx.suffix
            end
        end

        formattedoutput = split(sourcefilename, '/')[end] * " " * String(output .+ UInt8('0')) * '\n'
        write(outputfile, formattedoutput)

        println("done with $sourcefilename")
    end

    close(outputfile)
end

if length(ARGS) != 3
    println("usage: julia bin.jl <sourcefile> <stringsfile> <outputfile>")
    exit(1)
end

println("#$(Threads.nthreads()) we have all the threads")
match(ARGS...)
