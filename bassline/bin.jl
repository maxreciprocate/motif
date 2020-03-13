#!/usr/bin/env julia
include("./bassline.jl")

if length(ARGS) != 3
    println("usage: julia bin.jl <sourcefile> <stringsfile> <outputfile>")
    exit(1)
end

println("We have this many threads = $(Threads.nthreads())")
match(ARGS...)
