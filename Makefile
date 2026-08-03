.PHONY: main tDLGM util lint shampoo

_: main tDLGM util shampoo 

main:
	python -m tdlgm.main

shampoo:
	python -m tdlgm.main --shampoo_code true

tDLGM:
	python -m tdlgm.tDLGM

util:
	python -m tdlgm.util

lint:
	./lint.sh
