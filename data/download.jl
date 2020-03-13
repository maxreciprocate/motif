using ProgressMeter

if !isdir("bank")
    mkdir("bank")
end

ids = readlines("all_ids.csv")
p = Progress(length(ids))

@Threads.threads for id in ids
    file = "bank/pseudo$id.fasta.gz"
    link = "http://1001genomes.org/data/GMI-MPI/releases/v3.1/pseudogenomes/fasta/pseudo$id.fasta.gz"

    download(link, file)
    next!(p; showvalues = [(:downloaded, file)])
end
