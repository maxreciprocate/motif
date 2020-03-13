module Baseline

import Base.match
using DataStructures
using Test

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

    output = zeros(UInt8, length(strings))
    for sourcefilename in readlines(sourcesfilename)
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

        write(outputfile, String(output .+ UInt8('0')))
        write(outputfile, '\n')
        println("done with $sourcefilename")

        fill!(output, 0)
    end

    close(outputfile)
end

export match
end
