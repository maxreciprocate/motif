jam: jam.o jam.h main.cc
	g++ -O3 -L/opt/cuda/lib64 -I/opt/cuda/include -lcuda -lcudart -lpthread jam.o jam.h main.cc -o jam

jam.o: jam.cu jam.h
	nvcc -Xptxas -O3,-v jam.cu -c

test: jam
	../test.jl jam

iter: iter-1
iter-%: jam
	nodemon -x "make && ./jam ../data/$*genomes.txt ../data/8000000markers.csv out.txt" -e "cc,cu,h"

iter-test:
	nodemon -x "make test" -e "cc,cu,h"

bench-%: jam
	time -f "$*,8M: %E" ./jam ../data/$*genomes.txt ../data/8000000markers.csv out.txt

clean: jam
	rm jam jam.o
