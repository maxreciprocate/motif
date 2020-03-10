import Base.match
using DataStructures
using Test

mutable struct Vertex
    children::Dict{Char, Vertex}
    string::Union{String, Nothing}
    suffix::Union{Vertex, Nothing}
    isroot::Bool

    Vertex() = new(Dict{Char, Vertex}(), nothing, nothing, false)
end

function Base.show(io::IO, vx::Vertex)
    write(io, "Vertex{ $(keys(vx.children))$(if vx.string != nothing vx.string else "" end) }")
end

function add!(vx::Vertex, string::String, idx::Int=1)
    if idx > length(string)
        vx.string = string
    else
        char = string[idx]
        haskey(vx.children, char) || (vx.children[char] = Vertex())

        add!(vx.children[char], string, idx+1)
    end
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


function search(vx::Vertex, key::String)
    out = []

    for char in key
        vx = search(vx, char)
        evx = vx

        while true
            if evx.string != nothing
                push!(out, evx.string)
            end

            evx.isroot && break
            evx = evx.suffix
        end
    end

    out
end

function match(source::String, strings::String...)
    fsm = Vertex()
    fsm.isroot = true

    for string in strings
        add!(fsm, string)
    end

    build!(fsm)

    search(fsm, source)
end

@testset "trivial, single matches" begin
    matches = match("0", "0")
    @test length(matches) == 1
    @test "0" in matches

    matches = match("10", "1")
    @test length(matches) == 1
    @test "1" in matches

    matches = match("10111", "1")
    @test length(matches) == 4
    @test "1" in matches
end

@testset "matching multiple single-char strings" begin
    matches = match("10001", "0", "1", "3")
    @test length(matches) == 5
    @test "0" in matches
    @test "1" in matches
    @test "3" ∉ matches

    matches = match("0155555592", "0", "1", "9", "2")
    @test length(matches) == 4
    @test "0" in matches
    @test "1" in matches
    @test "9" in matches
    @test "2" in matches
end

@testset "matching multiple short intersecting strings" begin
    strings = ["123", "12345"]
    matches = match("12345", strings...)
    @test length(matches) == 2

    for s in strings
        @test s in matches
    end

    strings = ["123", "12345", "234"]
    matches = match("12345", strings...)
    @test length(matches) == 3

    for s in strings
        @test s in matches
    end

    strings = ["123", "12345", "23"]
    matches = match("12345", strings...)
    @test length(matches) == 3

    for s in strings
        @test s in matches
    end
end

@testset "intersections within intersections" begin
    matches = match("12345", "234", "3")
    @test length(matches) == 2

    matches = match("1234567890", "12345", "567890", "123", "890", "9")
    @test length(matches) == 5

    matches = match("1234567890", "12345", "567890", "123", "890", "345", "23", "12")
    @test length(matches) == 7

    matches = match("1234567890", "2345678", "34567", "456", "5")
    @test length(matches) == 4
end

@testset "find some intersects" begin
    to_match = ["GACTG", "ACT", "ACGACTGAGCACT"]

    matches = match("ATGTTGGACACTCGGCGGACGACTGAGCACTGGAACTTTTTAAA", to_match...)
    @test length(matches) == 6

    for p in to_match
        @test p in matches
    end
end

function bulktest(source::String, strings::Vector{String}, nmatches::Int)
    matches = match(source, strings...)
    @test length(matches) == nmatches

    for s in strings
        @test s in matches
    end
end

@testset "matching many little strings" begin
    bulktest("ATGTTGGACACTCGGCGGACGACTGAGCACTGGAACTTTTTAAA", ["ACGACTGAGCACT", "GAC", "ACG", "ACGACT", "ACT"], 10)
    bulktest("GGCTCAAATTACACGTAAACTTAAGAATATTCG", ["TAC", "ACA", "TTACA", "ATA", "TAT", "ATAT"] , 6)
    bulktest("CAGTCATAAAATACATTCAAGTATCAATAAATAG", ["CAGTCATAA", "AGT", "CAT", "GTCAT"], 6)
    bulktest("ACGACTGAGCACT", ["GACTG", "ACT"], 3)
    bulktest("ATGTTGGACACTCGGCGGACGACTGAGCACTGGAACTTTTTAAA", ["ACT", "ACGACTGAGCACT", "GACTG"], 6)
    bulktest("CACCCACAATCAGGCGAAGAGCCCCGAC", ["CAC", "GAG", "GCC"], 4)
end

@testset "small and big string on a moderate size source" begin
    source = open("drosophila_suzukii_rna.fa") do file
        read(file, String)
    end

    string = "ATGTTGGAACCGGCGGGGAACTTTTTAAATTTTAATGGCTTCAGCGAACCCGAAAAAGCACTCGAGGGTGCCATCATAAGAGAGATTGAAGATGGAGTTCGCTGTGAGCAATGTAAATCAGATTGCCCGGGTTTTGCAGCTCACGATTGGAGGAAAACCTGCCAATCCTGCAAATGTCCTCGCGAGGCACATGCCATATACCAGCAACAAACGACCAACGTCCACGAGCGACTCGGCTTCAAACTGGTTTCCCCGGCGGATTCCGGAGTGGAGGCGAGGGATCTGGGCTTCACGTGGGTTCCGCCCGGACTGCGAGCCTCGTCGCGGATCATCCGCTATTTCGAGCAGCTGCCCGATGAGGCGGTGCCCCGGTTGGGCAGCGAGGGAGCCTGCAGTCGGGAGCGCCAGATCTCGTACCAGCTGCCCAAACAGGACCTCTCGCTGGAGCACTGTAAGCACCTGGAGGTGCAGCACGAGTCCTCCTTCGAGGACTTTGTGACGGCGCGGAACGAAATCGCACTGGATATAGCCTACATCAAGGATGCACCCTACGATGAGCATTGTGCGCACTGTGATAACGAGATAGCTGCCGGCGAGCTGGTTGTAGCGGCGCCCAAGTTTGTGGAGAGCGTGATGTGGCACCCCAAGTGCTTCACCTGCAGCACCTGCAACCTGCTCCTGGTGGACCTCACCTACTGTGTCCACGACGACAAGGTCTACTGCGAGCGCCACTATGCGGAAATGCTGAAGCCCCGCTGCGCTGGCTGTGATGAGGTGAGTTCCCTCTAG"

    @time matches = match(source, "TAAAAAACGT")
    @test length(matches) == 39
    @test "TAAAAAACGT" in matches

    @time matches = match(source, "TAAAAAACGT", string)
    @test length(matches) == 40
    @test "TAAAAAACGT" in matches
    @test string in matches
end


@testset "match three big intersecting strings" begin
    source = open("drosophila_suzukii_rna.fa") do file
        read(file, String)
    end

    strings = ["ATGTTGGAACCGGCGGGGAACTTTTTAAATTTTAATGGCTTCAGCGAACCCGAAAAAGCACTCGAGGGTGCCATCATAAGAGAGATTGAAGATGGAGTTCGCTGTGAGCAATGTAAATCAGATTGCCCGGGTTTTGCAGCTCACGATTGGAGGAAAACCTGCCAATCCTGCAAATGTCCTCGCGAGGCACATGCCATATACCAGCAACAAACGACCAACGTCCACGAGCGACTCGGCTTCAAACTGGTTTCCCCGGCGGATTCCGGAGTGGAGGCGAGGGATCTGGG",
               "ATGTTGGAACCGGCGGGGAACTTTTTAAATTTTAATGGCTTCAGCGAACCCGAAAAAGCACTCGAGGGTGCCATCATAAGAGAGATTGAAGATGGAGTTCGCTGTGAGCAATGTAAATCAGATTGCCCGGGTTTTGCAGCTCACGATTGGAGGAAAACCTGCCAATCCTGCAAATGTCCTCGCGAGGCACATGCCATATACCAGCAACAAACGACCAACGTCCACGAGCGACTCGGCTTCAAACTGGTTTCCCCGGCGGATTCCGGAGTGGAGGCGAGGGATCTGGGCTTCACGTGGGTTCCGCCCGGACTGCGAGCCTCGTCGCGGATCATCCGCTATTTCGAGCAGCTGCCCGATGAGGCGGTGCCCCGGTTGGGCAGCGAGGGAGCCTGCAGTCGGGAGCGCCAGATCTCGTACCAGCTGCCCAAACAGGACCTCTCGCTGGAGCACTGTAAGCACCTGGAGGTGCAGCACGAGTCCTCCTTCGAGGACTTTGTGACGGCGCGGAACGAAATCGCACTGGATATAGCCTACATCAAGGATGCACCCTACGATGAGCATTGTGCGCACTGTGATAACGAGATAGCTGCCGGCGAGCTGGTTGTAGCGGCGCCCAAGTTTGTGGAGAGCGTGATGTGGCACCCCAAGTGCTTCACCTGCAGCACCTGCAACCTGCTCCTGGTGGACCTCACCTACTGTGTCCACGACGACAAGGTCTACTGCGAGCGCCACTATGCGGAAATGCTGAAGCCCCGCTGCGCTGGCTGTGATGAGGTGAGTTCCCTCTAG",
               "AATGGCTTCAGCGAACCCGAAAAAGCACTCGAGGGTGCCATCATAAGAGAGATTGAAGATGGAGTTCGCTGTGAGCAATGTAAATCAGATTGCCCGG"]

    @time matches = match(source, strings...)
    @test length(matches) == 3
end
