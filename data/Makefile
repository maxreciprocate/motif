MARKERS = 8000markers.csv 80000markers.csv 800000markers.csv 8000000markers.csv
GENOMES = 1genomes.txt 10genomes.txt 100genomes.txt 1000genomes.txt
SMALL = $(filter-out 1000genomes.txt, $(GENOMES))
VERYSMALL = $(filter-out 100genomes.txt, $(SMALL))

data-10: $(VERYSMALL) $(MARKERS)
data-100: $(SMALL) $(MARKERS)

all: $(GENOMES) $(MARKERS)

bank-%:
	JULIA_NUM_THREADS=32 julia download.jl $*

%genomes.txt: bank-%
	./extract.sh $*

%markers.csv:
	head -n $* markers.csv > $*markers.csv

clean:
	rm $(GENOMES) $(MARKERS)
