all: motif
motif:
	pip3 install --user --upgrade ./motif/

bench-%: data-% jam groove
	julia benchmark.jl $*

jam:
	$(MAKE) clean -C jam
	$(MAKE) -C jam

groove:
	$(MAKE) clean -C motif
	$(MAKE) -C motif

data-%:
	$(MAKE) $@ -C data

test_python_interface:
	GENOMES_LIST_PATH='./data/939genomes.txt' TESTS_DIR_PATH='../test_cases' python3 python_interface/test.py
