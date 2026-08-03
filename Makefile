.PHONY: main tDLGM util lint shampoo

_: main tDLGM util shampoo 


baseline:
	python -m tdlgm.main --verbose --baseline lstm

tune:
	python -m tdlgm.main --verbose --tune

main:
	python -m tdlgm.main --verbose --reduced_dataset 0.005

shampoo:
	python -m tdlgm.main --shampoo_code true --verbose 

tDLGM:
	python -m tdlgm.tDLGM --verbose 

util:
	python -m tdlgm.util --verbose 

lint:
	./lint.sh
