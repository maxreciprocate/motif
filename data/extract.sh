#!/usr/bin/bash
# usage: ./extract.sh <number_of_genomes>
if (( $# != 1)); then
  echo "usage: ./extract.sh <number_of_genomes>"
  exit 1
fi

if [[ ! -d bank ]]; then
  echo "run download.jl first"
  exit 1
fi

if [[ -e $1genomes.txt ]]; then
    rm $1genomes.txt
fi

# extracting all the boys (sorry)
find bank -name "*.fasta.gz" -exec gzip -d {} \;

genomes=$(ls bank | grep ".*\.fasta$" | head -n $1)

for genome in $genomes
do
  genome=bank/$genome

  # skip already cleansed genomes
  if [[ $(wc -l $genome | cut -d" " -f1) -eq 0 ]]; then
    echo "adding $genome"
    echo $genome >> $1genomes.txt
    continue
  fi

  awk -f clean.awk $genome > $genome.cleaned

  if [[ $? -eq 0 ]]; then
    mv $genome.cleaned $genome
    echo "adding $genome"
    echo $genome >> $1genomes.txt
  else
    rm $genome.cleaned
  fi
done
