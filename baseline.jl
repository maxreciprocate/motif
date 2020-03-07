import Base.match
using Test

mutable struct Vertex
    children::Dict{Char, Vertex}
    key::Union{String, Nothing}
    suffix::Union{Vertex, Nothing}
    # used for fsm's memoization
    char::Union{Char, Nothing}

    Vertex() = (
        vx = new();
        vx.children = Dict{Char, Vertex}();
        vx.key = nothing;
        vx.char = nothing;
        vx.suffix = vx;
    )

    Vertex(char::Char) = new(Dict{Char, Vertex}(), nothing, nothing, char)
end

Base.show(io::IO, vx::Vertex) = write(io, "Vertex{$(vx.key) -> $(keys(vx.children)) [$(vx.char)]}")
# ■

function add(vx::Vertex, key::String, idx::Int=1)
    if idx > length(key)
        vx.key = key
    else
        char = key[idx]
        haskey(vx.children, char) || (vx.children[char] = Vertex(char))

        add(vx.children[char], key, idx+1)
    end
end

function build(vx::Vertex)
    for (char, child) in vx.children
        if vx.suffix == vx
            child.suffix = vx
        else
            child.suffix = search(vx.suffix, child.char)
        end
    end

    for child in values(vx.children)
        build(child)
    end
end

function search(vx::Vertex, char::Char)
    if haskey(vx.children, char)
        vx.children[char]
    elseif vx.suffix == vx
        vx
    else
        search(vx.suffix, char)
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
