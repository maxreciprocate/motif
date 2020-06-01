#!/usr/bin/env julia
using Test

if length(ARGS) != 1
    println("usage: julia test.jl <path to an executable>")
    exit(1)
end

function match(source::String, markers::Vector{String})
    sourcesfn, fsources = mktemp()
    sourcefn, fsource = mktemp()
    markersfn, fmarkers = mktemp()
    outputfn, foutput = mktemp()

    write(fsource, source)
    write(fsources, last(split(sourcefn, '/')))

    close(foutput)
    close(fsource)
    close(fsources)

    for (idx, marker) in enumerate(markers)
        write(fmarkers, "$idx,$marker\n")
    end

    close(fmarkers)

    run(`./$(ARGS[1]) $sourcesfn $markersfn $outputfn`)

    return last(split(readline(outputfn), ' '))
end

function read_markers(source::String)
    return map((line) -> line[findfirst(isequal(','), line) + 1:length(line)], readlines(source))
end

function read_result(source::String)
    result = readline(source)
    return result[findfirst(isequal(' '), result) + 1: length(result)]
end

try
    match("A", ["A"])
catch e
    println(e)
    exit(1)
end

@testset "*" begin
    @testset "single char markers" begin
        @test match("A", ["A"]) == "1"
        @test match("A", ["C"]) == "0"
        @test match("AC", ["A"]) == "1"
        @test match("AA", ["A"]) == "1"
        @test match("AA", ["C"]) == "0"
        @test match("AC", ["C"]) == "1"
        @test match("ACAAA", ["G"]) == "0"
    end

    @testset "matching multiple single-char strings" begin
        @test match("ACCCA", ["A", "C", "G"]) == "110"
        @test match("ACTTTTTTCG", ["A", "C", "G"]) == "111"
        @test match("GCTTTTTTCG", ["T", "G", "C", "A"]) == "1110"
    end
    @testset "handling breaks" begin
        @test match("ACNNNA", ["ACA", "AC"]) == "01"
        @test match("NNNNNN", ["ACA", "AC"]) == "00"
        @test match("NNNNNAGTNNNNA", ["AT", "AGTA", "TA"]) == "000"
    end

    @testset "matching multiple short intersecting strings" begin
        @test match("ACGTA", ["ACG", "CGTA"]) == "11"
        @test match("ACGTA", ["AC", "CG", "GTA"]) == "111"
    end

    @testset "intersections within intersections" begin
        @test match("ACTGA", ["CTG", "T"]) == "11"

        matches = match("ACGTGTCACGT", ["ACGTG", "CACGT", "GTG", "TCA", "T", "TT"])
        @test matches == "111110"

        matches = match("ACGTGTCACGT", ["TCACC", "CGTA", "CACGT", "CAC", "ACGT"])
        @test matches == "00111"

        matches = match("ACGTGTCACGT", ["GC", "TGTCG", "GTGTCA", "GTC"])
        @test matches == "0011"
    end

    @testset "intersection within larger string" begin
        @test match("ATGTTGGACACTCGGCGGACGACTGAGCACTGGAACTTTTTAAA", ["GACTG", "ACT","ACGACTGAGCACT"]) == "111"
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
        bulktest("AAAAAAAAAA", ["A", "AA", "AAA", "AAAA", "AAAAA", "AAAAAA"], "111111")
    end

    @testset "with repeating markers" begin
        @test match("ACCCA", ["A", "C", "G", "C", "C", "C"]) == "110111"
        @test match("ACCCA", ["ACCCA", "ACCCA", "A", "ACCA", "TG"]) == "11100"
        @test match("GGGGG", ["GG", "GG", "GGG", "GGGGG", "GGGGG", "GGGGGG"]) == "111110"
        @test match("AGGGGGANA", ["GGGG", "AG", "GAA", "GA", "GGGG", "GAA", "GGGG"]) == "1101101"
    end

    # test_genome = "data/bank/pseudo139.fasta"
    # test_markers_set = "data/8000000markers.csv"
    # test_results_set = "tests/data/result8000000.txt"

    # @testset "real genome and real markers" begin
    #     @test match(readline(test_genome), read_markers(test_markers_set)) == read_result(test_results_set)
    # end
end
