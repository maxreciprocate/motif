using ProgressMeter

if !isdir("bank")
    mkdir("bank")
end

ids = readlines("all_ids.csv")

if length(ARGS) > 0
    try
        ngenomes = parse(Int, ARGS[1])
        global ids = ids[1:ngenomes]
    catch
        println("usage: julia download.jl <number of genomes>")
        exit(1)
    end
end

p = Progress(length(ids))

@Threads.threads for id in ids
    file = "bank/pseudo$id.fasta.gz"
    link = "http://1001genomes.org/data/GMI-MPI/releases/v3.1/pseudogenomes/fasta/pseudo$id.fasta.gz"

    # skip, if the file or it's extraction is present
    if isfile(file) || isfile(reverse(reverse(file)[4:end]))
        println("skipping $file")
        continue
    end

    try
        download(link, file)
    catch
        isfile(file) && rm(file)
        println("cannot download $file")
    end

    next!(p; showvalues = [(:downloaded, file)])
end
