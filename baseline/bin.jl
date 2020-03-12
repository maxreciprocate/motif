#!/usr/bin/env julia
include("./baseline.jl")
using .Baseline

if length(ARGS) != 3
    println("usage: julia bin.jl <sourcefile> <stringsfile> <outputfile>")
    exit(1)
end

match(ARGS...)
