module Baseline

export match

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

function Base.show(io::IO, vx::Vertex)
    write(io, "Vertex{ $(keys(vx.children))$(if vx.string != nothing vx.string else "" end) }")
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

function match(sourcefilename::String, stringsfilename::String, outputfilename::String)
    source = open(sourcefilename) do file
        read(file, String)
    end

    markers = map(line -> split(line, ',')[2], readlines(stringsfilename))

    output = match(source, markers)

    open(outputfilename, "w+") do file
        write(file, output)
    end

    output
end

end
