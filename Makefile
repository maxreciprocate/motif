all: bench-10

bench-%: data-%
	julia benchmark.jl $*

data-%:
	$(MAKE) $@ -C data
