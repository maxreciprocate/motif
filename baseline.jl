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

# @testset "matching multiple short intersecting strings" begin
#     strings = ["123", "12345"]
#     matches = match("12345", strings...)
#     @test length(matches) == 2

#     for s in strings
#         @test s in matches
#     end

#     strings = ["123", "12345", "234"]
#     matches = match("12345", strings...)
#     @test length(matches) == 3

#     for s in strings
#         @test s in matches
#     end

#     strings = ["123", "12345", "23"]
#     matches = match("12345", strings...)
#     @test length(matches) == 3

#     for s in strings
#         @test s in matches
#     end
# end

# @testset "intersections within intersections" begin
#     matches = match("12345", "234", "3")
#     @test length(matches) == 2

#     matches = match("1234567890", "12345", "567890", "123", "890", "9")
#     @test length(matches) == 5

#     matches = match("1234567890", "12345", "567890", "123", "890", "345", "23", "12")
#     @test length(matches) == 7

#     matches = match("1234567890", "2345678", "34567", "456", "5")
#     @test length(matches) == 4
# end

# @testset "find some intersects" begin
#     to_match = ["GACTG", "ACT", "ACGACTGAGCACT"]

#     matches = match("ATGTTGGACACTCGGCGGACGACTGAGCACTGGAACTTTTTAAA", to_match...)
#     @test length(matches) == 6

#     for p in to_match
#         @test p in matches
#     end
# end

# function bulktest(source::String, strings::Vector{String}, nmatches::Int)
#     matches = match(source, strings...)
#     @test length(matches) == nmatches

#     for s in strings
#         @test s in matches
#     end
# end

# @testset "matching many little strings" begin
#     bulktest("ATGTTGGACACTCGGCGGACGACTGAGCACTGGAACTTTTTAAA", ["ACGACTGAGCACT", "GAC", "ACG", "ACGACT", "ACT"], 10)
#     bulktest("GGCTCAAATTACACGTAAACTTAAGAATATTCG", ["TAC", "ACA", "TTACA", "ATA", "TAT", "ATAT"] , 6)
#     bulktest("CAGTCATAAAATACATTCAAGTATCAATAAATAG", ["CAGTCATAA", "AGT", "CAT", "GTCAT"], 6)
#     bulktest("ACGACTGAGCACT", ["GACTG", "ACT"], 3)
#     bulktest("ATGTTGGACACTCGGCGGACGACTGAGCACTGGAACTTTTTAAA", ["ACT", "ACGACTGAGCACT", "GACTG"], 6)
#     bulktest("CACCCACAATCAGGCGAAGAGCCCCGAC", ["CAC", "GAG", "GCC"], 4)
#     bulktest("1111111111", "1", "11", "111", "1111", "11111", "111111", 45)
# end

# @testset "small and big string on a moderate size source" begin
#     source = open("./data/drosophila.fasta") do file
#         read(file, String)
#     end

#     string = "ATGTTGGAACCGGCGGGGAACTTTTTAAATTTTAATGGCTTCAGCGAACCCGAAAAAGCACTCGAGGGTGCCATCATAAGAGAGATTGAAGATGGAGTTCGCTGTGAGCAATGTAAATCAGATTGCCCGGGTTTTGCAGCTCACGATTGGAGGAAAACCTGCCAATCCTGCAAATGTCCTCGCGAGGCACATGCCATATACCAGCAACAAACGACCAACGTCCACGAGCGACTCGGCTTCAAACTGGTTTCCCCGGCGGATTCCGGAGTGGAGGCGAGGGATCTGGGCTTCACGTGGGTTCCGCCCGGACTGCGAGCCTCGTCGCGGATCATCCGCTATTTCGAGCAGCTGCCCGATGAGGCGGTGCCCCGGTTGGGCAGCGAGGGAGCCTGCAGTCGGGAGCGCCAGATCTCGTACCAGCTGCCCAAACAGGACCTCTCGCTGGAGCACTGTAAGCACCTGGAGGTGCAGCACGAGTCCTCCTTCGAGGACTTTGTGACGGCGCGGAACGAAATCGCACTGGATATAGCCTACATCAAGGATGCACCCTACGATGAGCATTGTGCGCACTGTGATAACGAGATAGCTGCCGGCGAGCTGGTTGTAGCGGCGCCCAAGTTTGTGGAGAGCGTGATGTGGCACCCCAAGTGCTTCACCTGCAGCACCTGCAACCTGCTCCTGGTGGACCTCACCTACTGTGTCCACGACGACAAGGTCTACTGCGAGCGCCACTATGCGGAAATGCTGAAGCCCCGCTGCGCTGGCTGTGATGAGGTGAGTTCCCTCTAG"

#     @time matches = match(source, "TAAAAAACGT")
#     @test length(matches) == 39
#     @test "TAAAAAACGT" in matches

#     @time matches = match(source, "TAAAAAACGT", string)
#     @test length(matches) == 40
#     @test "TAAAAAACGT" in matches
#     @test string in matches
# end


# @testset "match three big intersecting strings" begin
#     source = open("./data/drosophila.fasta") do file
#         read(file, String)
#     end

#     strings = ["ATGTTGGAACCGGCGGGGAACTTTTTAAATTTTAATGGCTTCAGCGAACCCGAAAAAGCACTCGAGGGTGCCATCATAAGAGAGATTGAAGATGGAGTTCGCTGTGAGCAATGTAAATCAGATTGCCCGGGTTTTGCAGCTCACGATTGGAGGAAAACCTGCCAATCCTGCAAATGTCCTCGCGAGGCACATGCCATATACCAGCAACAAACGACCAACGTCCACGAGCGACTCGGCTTCAAACTGGTTTCCCCGGCGGATTCCGGAGTGGAGGCGAGGGATCTGGG",
#                "ATGTTGGAACCGGCGGGGAACTTTTTAAATTTTAATGGCTTCAGCGAACCCGAAAAAGCACTCGAGGGTGCCATCATAAGAGAGATTGAAGATGGAGTTCGCTGTGAGCAATGTAAATCAGATTGCCCGGGTTTTGCAGCTCACGATTGGAGGAAAACCTGCCAATCCTGCAAATGTCCTCGCGAGGCACATGCCATATACCAGCAACAAACGACCAACGTCCACGAGCGACTCGGCTTCAAACTGGTTTCCCCGGCGGATTCCGGAGTGGAGGCGAGGGATCTGGGCTTCACGTGGGTTCCGCCCGGACTGCGAGCCTCGTCGCGGATCATCCGCTATTTCGAGCAGCTGCCCGATGAGGCGGTGCCCCGGTTGGGCAGCGAGGGAGCCTGCAGTCGGGAGCGCCAGATCTCGTACCAGCTGCCCAAACAGGACCTCTCGCTGGAGCACTGTAAGCACCTGGAGGTGCAGCACGAGTCCTCCTTCGAGGACTTTGTGACGGCGCGGAACGAAATCGCACTGGATATAGCCTACATCAAGGATGCACCCTACGATGAGCATTGTGCGCACTGTGATAACGAGATAGCTGCCGGCGAGCTGGTTGTAGCGGCGCCCAAGTTTGTGGAGAGCGTGATGTGGCACCCCAAGTGCTTCACCTGCAGCACCTGCAACCTGCTCCTGGTGGACCTCACCTACTGTGTCCACGACGACAAGGTCTACTGCGAGCGCCACTATGCGGAAATGCTGAAGCCCCGCTGCGCTGGCTGTGATGAGGTGAGTTCCCTCTAG",
#                "AATGGCTTCAGCGAACCCGAAAAAGCACTCGAGGGTGCCATCATAAGAGAGATTGAAGATGGAGTTCGCTGTGAGCAATGTAAATCAGATTGCCCGG"]

#     @time matches = match(source, strings...)
#     @test length(matches) == 3
# end

@testset "benchmark 2000 markers" begin
    @time match("data/thaliana.fna", "data/2000markers.csv", "result.txt")
end
