.PHONY: main tDLGM util eval lint shampoo

_: main tDLGM util shampoo 


tune:
	python -m tdlgm.main --verbose --tune

main:
	python -m tdlgm.main --verbose 

shampoo:
	python -m tdlgm.main --shampoo_code true --verbose 

tDLGM:
	python -m tdlgm.tDLGM --verbose 

util:
	python -m tdlgm.util --verbose 

eval:
	python -m tdlgm.eval --verbose 

lint:
	./lint.sh
