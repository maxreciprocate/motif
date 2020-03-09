import Base.match
using Test

FIXED = false

mutable struct Vertex
    children::Dict{Char, Vertex}
    key::Union{String, Nothing}
    link::Union{Vertex, Nothing}
    isroot::Bool

    Vertex() = new(Dict{Char, Vertex}(), nothing, nothing, false)
end

Base.show(io::IO, vx::Vertex) = write(io, "Vertex{ $(keys(vx.children))$(if vx.key != nothing vx.key else "" end) }")
# ■

function add(vx::Vertex, key::String, idx::Int=1)
    if idx > length(key)
        vx.key = key
    else
        char = key[idx]
        haskey(vx.children, char) || (vx.children[char] = Vertex())

        add(vx.children[char], key, idx+1)
    end
end

function build(vx::Vertex)
    for (char, child) in vx.children
        if vx.isroot
            child.link = vx
        elseif vx.link != nothing
            child.link = search(vx.link, char)
        end
    end

    for child in values(vx.children)
        build(child)
    end
end

function search(vx::Vertex, char::Char)
    if haskey(vx.children, char)
        vx.children[char]
    elseif vx.link != nothing
        search(vx.link, char)
    else
        vx
    end
end

function search(vx::Vertex, key::String)
    xs = []

    for char in key
        vx = search(vx, char)
        vx.key == nothing || push!(xs, vx.key)
    end

    xs
end

function match(dictionary::Vector{String}, source::String)
    fsm = Vertex()
    fsm.isroot = true

    for line in dictionary
        add(fsm, line)
    end

    build(fsm)

    search(fsm, source)
end

function test_intersections(test_name::String ,str::String, to_match::Vector{String}, n_matches::Int)
    @testset "$test_name" begin
        
    @time matches = match(to_match, str)
    @test_broken length(matches) == n_matches  #for now broken

    for p in to_match
        if !FIXED @test_skip p in matches  #set FIXED = true when fixing this stuff
        else @test p in matches 
        end
    end

    end
    
end

# function get_rna(file_name::String)


# ■
@testset "trivial" begin
    matches = match(["0"], "0")
    @test length(matches) == 1
    @test "0" in matches

    matches = match(["1"], "10")
    @test length(matches) == 1
    @test "1" in matches

    matches = match(["1", "0"], "10")
    @test length(matches) == 2
    @test "0" in matches
    @test "1" in matches
end

@testset "drosophila simple" begin
    rna = open("drosophila_suzukii_rna.fa") do file
        read(file, String)
    end

    @time matches = match(["TAAAAAACGT"], rna)
    @test length(matches) == 39
    @test "TAAAAAACGT" in matches

    section = "ATGTTGGAACCGGCGGGGAACTTTTTAAATTTTAATGGCTTCAGCGAACCCGAAAAAGCACTCGAGGGTGCCATCATAAGAGAGATTGAAGATGGAGTTCGCTGTGAGCAATGTAAATCAGATTGCCCGGGTTTTGCAGCTCACGATTGGAGGAAAACCTGCCAATCCTGCAAATGTCCTCGCGAGGCACATGCCATATACCAGCAACAAACGACCAACGTCCACGAGCGACTCGGCTTCAAACTGGTTTCCCCGGCGGATTCCGGAGTGGAGGCGAGGGATCTGGGCTTCACGTGGGTTCCGCCCGGACTGCGAGCCTCGTCGCGGATCATCCGCTATTTCGAGCAGCTGCCCGATGAGGCGGTGCCCCGGTTGGGCAGCGAGGGAGCCTGCAGTCGGGAGCGCCAGATCTCGTACCAGCTGCCCAAACAGGACCTCTCGCTGGAGCACTGTAAGCACCTGGAGGTGCAGCACGAGTCCTCCTTCGAGGACTTTGTGACGGCGCGGAACGAAATCGCACTGGATATAGCCTACATCAAGGATGCACCCTACGATGAGCATTGTGCGCACTGTGATAACGAGATAGCTGCCGGCGAGCTGGTTGTAGCGGCGCCCAAGTTTGTGGAGAGCGTGATGTGGCACCCCAAGTGCTTCACCTGCAGCACCTGCAACCTGCTCCTGGTGGACCTCACCTACTGTGTCCACGACGACAAGGTCTACTGCGAGCGCCACTATGCGGAAATGCTGAAGCCCCGCTGCGCTGGCTGTGATGAGGTGAGTTCCCTCTAG"

    @time matches = match([section], rna)
    @test length(matches) == 1
    @test section in matches
end

@testset "find ACT" begin
    pattern = "ACT"
    to_match = [pattern]
    @time matches = match(to_match, "ACGTACTGAGCACT")
    @test length(matches) == 2
    @test pattern in matches

    @time matches = match(to_match, "ACGTAGTGAGCAC")
    @test length(matches) == 0

    @time matches = match(to_match, "ATGTTGGACACTCGGCGGGGAACTTTTTAAA")
    @test length(matches) == 2
    @test pattern in matches
end

@testset "find GAG" begin
    pattern = "GAG"
    to_match = [pattern]
    @time matches = match(to_match, "ACGTACTGAGCACT")
    @test length(matches) == 1
    @test pattern in matches

    @time matches = match(to_match, "ACGTAGTGAGCAC")
    @test length(matches) == 1
    @test pattern in matches

    @time matches = match(to_match, "ATGTTGGACACTCGGCGGACGACTGAGCACTGGAACTTTTTAAA")
    @test length(matches) == 1
    # @test pattern in matches
end

@testset "find GAG, ACT, GTA" begin
    to_match = ["GACTG", "ACT"]
    @time matches = match(to_match, "ACGACTGAGCACT")
    @test_broken length(matches) == 3
    
    for p in to_match
        @test p in matches
    end
end

@testset "find some intersects" begin 
    to_match = ["GACTG", "ACT",  "ACGACTGAGCACT"]

    @time matches = match(to_match, "ATGTTGGACACTCGGCGGACGACTGAGCACTGGAACTTTTTAAA")
    @test_broken length(matches) == 6
    
    for p in to_match
        if p == "GACTG" @test_broken p in matches  # some stuff happening to GACTG - function can't match it
        else @test p in matches end
    end
end

test_intersections("intersection function", 
                    "ATGTTGGACACTCGGCGGACGACTGAGCACTGGAACTTTTTAAA",
                    ["ACGACTGAGCACT", "GAC", "ACG", "ACGACT", "ACT"], 10
                    )

test_intersections("more intersections",
                    "GGCTCAAATTACACGTAAACTTAAGAATATTCG",
                    ["TAC", "ACA", "TTACA", "ATA", "TAT", "ATAT"], 6
                    )

test_intersections("one more intersection test",
                   "CAGTCATAAAATACATTCAAGTATCAATAAATAG",
                  ["CAGTCATAA", "AGT", "CAT", "GTCAT"], 6
                  )

@testset "more intersections" begin
    
end

@testset "find CAC GAG GCC" begin
    str = "CACCCACAATCAGGCGAAGAGCCCCGAC"
    to_match = ["CAC", "GAG", "GCC"]
    @time matches = match(to_match, str)
    @test length(matches) == 4

    for p in to_match
        @test p in matches
    end
end

@testset "another teest" begin
    rna = open("drosophila_suzukii_rna.fa") do file
        read(file, String)
    end
    to_match = "TTCAGGAGAAGCTATGCTCGGACCCAGAAGTGGTAAACAAGGACCACTGCCACAATCTGGCCAATGAGCACGAGGCGCTTCTGGAGGACTGGTTCACCCACAATCAGGCGAAGAGCCCCGACCTGCAATCGTGGCTTTGCATCGACCAGCTGGCCGTCTGTTGTCCGCCGAACACCTACGGAGCAGATTGCCAGCCCTGCACCGACTGCAGCGGAAACGGTAAATGCAAGGGAGCCGGTACTCGAAAAGGCAACGGAAAGTGCAAATGTGATCCTGGCTATGCGGGACCCAACTGCAATGAGTGTGGATCGATGCACTACGAGTCCTTCCGTGACGAGAA"
    @time matches = match([to_match], rna)
    # println("length = " , length(matches))
    @test length(matches) == 2
    @test to_match in matches
end

@testset "more complex test" begin
    rna = open("drosophila_suzukii_rna.fa") do file
        read(file, String)
    end
    section1 = "TTCAGGAGAAGCTATGCTCGGACCCAGAAGTGGTAAACAAGGACCACTGCCACAATCTGGCCAATGAGCACGAGGCGCTTCTGGAGGACTGGTTCACCCACAATCAGGCGAAGAGCCCCGACCTGCAATCGTGGCTTTGCATCGACCAGCTGGCCGTCTGTTGTCCGCCGAACACCTACGGAGCAGATTGCCAGCCCTGCACCGACTGCAGCGGAAACGGTAAATGCAAGGGAGCCGGTACTCGAAAAGGCAACGGAAAGTGCAAATGTGATCCTGGCTATGCGGGACCCAACTGCAATGAGTGTGGATCGATGCACTACGAGTCCTTCCGTGACGAGAA"
    section2 = "ATGTTGGAACCGGCGGGGAACTTTTTAAATTTTAATGGCTTCAGCGAACCCGAAAAAGCACTCGAGGGTGCCATCATAAGAGAGATTGAAGATGGAGTTCGCTGTGAGCAATGTAAATCAGATTGCCCGGGTTTTGCAGCTCACGATTGGAGGAAAACCTGCCAATCCTGCAAATGTCCTCGCGAGGCACATGCCATATACCAGCAACAAACGACCAACGTCCACGAGCGACTCGGCTTCAAACTGGTTTCCCCGGCGGATTCCGGAGTGGAGGCGAGGGATCTGGGCTTCACGTGGGTTCCGCCCGGACTGCGAGCCTCGTCGCGGATCATCCGCTATTTCGAGCAGCTGCCCGATGAGGCGGTGCCCCGGTTGGGCAGCGAGGGAGCCTGCAGTCGGGAGCGCCAGATCTCGTACCAGCTGCCCAAACAGGACCTCTCGCTGGAGCACTGTAAGCACCTGGAGGTGCAGCACGAGTCCTCCTTCGAGGACTTTGTGACGGCGCGGAACGAAATCGCACTGGATATAGCCTACATCAAGGATGCACCCTACGATGAGCATTGTGCGCACTGTGATAACGAGATAGCTGCCGGCGAGCTGGTTGTAGCGGCGCCCAAGTTTGTGGAGAGCGTGATGTGGCACCCCAAGTGCTTCACCTGCAGCACCTGCAACCTGCTCCTGGTGGACCTCACCTACTGTGTCCACGACGACAAGGTCTACTGCGAGCGCCACTATGCGGAAATGCTGAAGCCCCGCTGCGCTGGCTGTGATGAGGTGAGTTCCCTCTAG"

    @time matches = match([section1, section2], rna)
    @test length(matches) == 3

end