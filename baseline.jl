#!/usr/bin/env julia

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

# ■
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

# ■

@testset "trivial, single matches" begin
    @test match("0", ["0"]) == "1"
    @test match("0", ["1"]) == "0"
    @test match("10", ["1"]) == "1"
    @test match("11", ["1"]) == "1"
    @test match("11", ["0"]) == "0"
    @test match("10", ["0"]) == "1"
    @test match("10111", ["0"]) == "1"
end

@testset "matching multiple single-char strings" begin
    @test match("10001", ["0", "1", "3"]) == "110"
    @test match("0155555592", ["0", "1", "9", "2"]) == "1111"
    @test match("0155555592", ["7", "5", "3", "1"]) == "0101"
end

@testset "matching multiple short intersecting strings" begin
    strings = ["123", "12345"]
    @test match("12345", strings) == "11"

    strings = ["123", "12345", "234"]
    @test match("12345", strings) == "111"

    strings = ["123", "12345", "23"]
    @test match("12345", strings) == "111"
end

@testset "intersections within intersections" begin
    @test match("12345", ["234", "3"]) == "11"

    matches = match("1234567890", ["12345", "567890", "123", "890", "9"])
    @test matches == "11111"

    matches = match("1234567890", ["12345", "567890", "123", "890", "345", "23", "12"])
    @test matches == "1111111"

    matches = match("1234567890", ["2345678", "34567", "456", "5"])
    @test matches == "1111"

end

@testset "find some intersects" begin
    to_match = ["GACTG", "ACT", "ACGACTGAGCACT"]

    matches = match("ATGTTGGACACTCGGCGGACGACTGAGCACTGGAACTTTTTAAA", to_match)
    @test matches == "111"
end

function bulktest(source::String, strings::Vector{String}, match_result::String)
    @test match(source, strings) == match_result
end

@testset "matching many little strings" begin
    bulktest("ATGTTGGACACTCGGCGGACGACTGAGCACTGGAACTTTTTAAA", ["ACGACTGAGCACT", "GAC", "ACG", "ACGACT", "ACT"], "11111")
    bulktest("GGCTCAAATTACACGTAAACTTAAGAATATTCG", ["TAC", "ACA", "TTACA", "ATA", "TAT", "ATAT"] , "111111")
    bulktest("CAGTCATAAAATACATTCAAGTATCAATAAATAG", ["CAGTCATAA", "AGT", "CAT", "GTCAT"], "1111")
    bulktest("ACGACTGAGCACT", ["GACTG", "ACT"], "11")
    bulktest("ATGTTGGACACTCGGCGGACGACTGAGCACTGGAACTTTTTAAA", ["ACT", "ACGACTGAGCACT", "GACTG"], "111")
    bulktest("CACCCACAATCAGGCGAAGAGCCCCGAC", ["CAC", "GAG", "GCC"], "111")
    bulktest("1111111111", ["1", "11", "111", "1111", "11111", "111111"], "111111")
end

@testset "small and big string on a moderate size source" begin
    source = open("./data/drosophila.fasta") do file
        read(file, String)
    end

    string = "ATGTTGGAACCGGCGGGGAACTTTTTAAATTTTAATGGCTTCAGCGAACCCGAAAAAGCACTCGAGGGTGCCATCATAAGAGAGATTGAAGATGGAGTTCGCTGTGAGCAATGTAAATCAGATTGCCCGGGTTTTGCAGCTCACGATTGGAGGAAAACCTGCCAATCCTGCAAATGTCCTCGCGAGGCACATGCCATATACCAGCAACAAACGACCAACGTCCACGAGCGACTCGGCTTCAAACTGGTTTCCCCGGCGGATTCCGGAGTGGAGGCGAGGGATCTGGGCTTCACGTGGGTTCCGCCCGGACTGCGAGCCTCGTCGCGGATCATCCGCTATTTCGAGCAGCTGCCCGATGAGGCGGTGCCCCGGTTGGGCAGCGAGGGAGCCTGCAGTCGGGAGCGCCAGATCTCGTACCAGCTGCCCAAACAGGACCTCTCGCTGGAGCACTGTAAGCACCTGGAGGTGCAGCACGAGTCCTCCTTCGAGGACTTTGTGACGGCGCGGAACGAAATCGCACTGGATATAGCCTACATCAAGGATGCACCCTACGATGAGCATTGTGCGCACTGTGATAACGAGATAGCTGCCGGCGAGCTGGTTGTAGCGGCGCCCAAGTTTGTGGAGAGCGTGATGTGGCACCCCAAGTGCTTCACCTGCAGCACCTGCAACCTGCTCCTGGTGGACCTCACCTACTGTGTCCACGACGACAAGGTCTACTGCGAGCGCCACTATGCGGAAATGCTGAAGCCCCGCTGCGCTGGCTGTGATGAGGTGAGTTCCCTCTAG"

    @time matches = match(source, ["TAAAAAACGT", string])
    @test matches == "11"
end


@testset "match three big intersecting strings" begin
    source = open("./data/drosophila.fasta") do file
        read(file, String)
    end

    strings = ["ATGTTGGAACCGGCGGGGAACTTTTTAAATTTTAATGGCTTCAGCGAACCCGAAAAAGCACTCGAGGGTGCCATCATAAGAGAGATTGAAGATGGAGTTCGCTGTGAGCAATGTAAATCAGATTGCCCGGGTTTTGCAGCTCACGATTGGAGGAAAACCTGCCAATCCTGCAAATGTCCTCGCGAGGCACATGCCATATACCAGCAACAAACGACCAACGTCCACGAGCGACTCGGCTTCAAACTGGTTTCCCCGGCGGATTCCGGAGTGGAGGCGAGGGATCTGGG",
               "ATGTTGGAACCGGCGGGGAACTTTTTAAATTTTAATGGCTTCAGCGAACCCGAAAAAGCACTCGAGGGTGCCATCATAAGAGAGATTGAAGATGGAGTTCGCTGTGAGCAATGTAAATCAGATTGCCCGGGTTTTGCAGCTCACGATTGGAGGAAAACCTGCCAATCCTGCAAATGTCCTCGCGAGGCACATGCCATATACCAGCAACAAACGACCAACGTCCACGAGCGACTCGGCTTCAAACTGGTTTCCCCGGCGGATTCCGGAGTGGAGGCGAGGGATCTGGGCTTCACGTGGGTTCCGCCCGGACTGCGAGCCTCGTCGCGGATCATCCGCTATTTCGAGCAGCTGCCCGATGAGGCGGTGCCCCGGTTGGGCAGCGAGGGAGCCTGCAGTCGGGAGCGCCAGATCTCGTACCAGCTGCCCAAACAGGACCTCTCGCTGGAGCACTGTAAGCACCTGGAGGTGCAGCACGAGTCCTCCTTCGAGGACTTTGTGACGGCGCGGAACGAAATCGCACTGGATATAGCCTACATCAAGGATGCACCCTACGATGAGCATTGTGCGCACTGTGATAACGAGATAGCTGCCGGCGAGCTGGTTGTAGCGGCGCCCAAGTTTGTGGAGAGCGTGATGTGGCACCCCAAGTGCTTCACCTGCAGCACCTGCAACCTGCTCCTGGTGGACCTCACCTACTGTGTCCACGACGACAAGGTCTACTGCGAGCGCCACTATGCGGAAATGCTGAAGCCCCGCTGCGCTGGCTGTGATGAGGTGAGTTCCCTCTAG",
               "AATGGCTTCAGCGAACCCGAAAAAGCACTCGAGGGTGCCATCATAAGAGAGATTGAAGATGGAGTTCGCTGTGAGCAATGTAAATCAGATTGCCCGG"]

    @time matches = match(source, strings)
    @test matches == "111"
end

@testset "benchmark 2000 markers" begin
    @time match("./data/thaliana.fna", "./data/mark2.csv", "result.txt")
end
