all: bench-100

bench-%: data-% jam groove
	julia benchmark.jl $*

jam:
	$(MAKE) clean -C jam
	$(MAKE) -C jam

groove:
	$(MAKE) clean -C motif
	$(MAKE) -C motif

jam_lib:
	pip3 install --user ./jam_lib/ --upgrade

data-%:
	$(MAKE) $@ -C data

test_python_interface:
	GENOMES_LIST_PATH='./data/1000genomes.txt' TESTS_DIR_PATH='../test_cases' python3 python_interface/test.py