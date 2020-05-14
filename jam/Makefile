jam: jam.o jam.h main.cc
	g++-9 -O3 -lpthread jam.h main.cc -c
	nvcc -o jam main.o jam.o -lcudart -lcuda

jam.o:
	nvcc -Xptxas -O3,-v jam.cu -c

test: jam
	../test.jl jam

clean:
	rm jam jam.o
