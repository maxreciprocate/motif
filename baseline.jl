import Base.match
using Test

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
