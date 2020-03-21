genomes_path_file = "../data/10genomes.txt"
markers_file = "../data/80000markers.csv"
output_file = "./build/result.txt"

numbers_of_threads = "4"

all: main

main:
	mkdir -p build
	cd build; CMake -DCMAKE_BUILD_TYPE=Release ..; make
	./build/main $(genomes_path_file) $(markers_file) $(output_file) $(numbers_of_threads)

clean:
	rm -rf build