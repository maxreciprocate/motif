jam: jam.o jam.h main.cc
	g++ -O3 -L/opt/cuda/lib64 -I/opt/cuda/include -lcuda -lcudart -lpthread jam.o jam.h main.cc -o jam

jam.o:
	nvcc -Xptxas -O3,-v jam.cu -c

test: jam
	../test.jl jam

clean:
	rm jam jam.o
