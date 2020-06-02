all: bench-100

bench-%: data-% jam groove
	julia benchmark.jl $*

jam:
	$(MAKE) clean -C jam
	$(MAKE) -C jam

groove:
	$(MAKE) clean -C motif
	$(MAKE) -C motif

jam_lib: jam_lib/jam_lib.cpp jam_lib/jam/jam_run.cc jam_lib/jam/jam.cu jam_lib/jam/jam.h 
	pip3 install --user ./jam_lib/ --upgrade

data-%:
	$(MAKE) $@ -C data

test_python_interface:
	GENOMES_LIST_PATH='./data/939genomes.txt' TESTS_DIR_PATH='../test_cases' python3 python_interface/test.py
