using ProgressMeter

if !isdir("data/bank")
    mkdir("data/bank")
end

ids = open("data/all_ids.csv") do file
    readlines(file)
end

p = Progress(length(ids))

@Threads.threads for id in ids
    file = "data/bank/pseudo$id.fasta.gz"
    link = "http://1001genomes.org/data/GMI-MPI/releases/v3.1/pseudogenomes/fasta/pseudo$id.fasta.gz"

    download(link, file)
    next!(p; showvalues = [(:downloaded, file)])
end
