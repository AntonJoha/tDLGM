.PHONY: main tDLGM util lint shampoo

_: main tDLGM util shampoo 


tune:
	python -m tdlgm.main --tune --verbose

main:
	python -m tdlgm.main --verbose 

shampoo:
	python -m tdlgm.main --shampoo_code true --verbose 

tDLGM:
	python -m tdlgm.tDLGM --verbose 

util:
	python -m tdlgm.util --verbose 

lint:
	./lint.sh
