.PHONY: main tDLGM util lint shampoo

_: main tDLGM util shampoo baseline baseline_shampoo tune


baseline:
	python -m tdlgm.main --verbose --baseline lstm --reduced_dataset 0.001
	

baseline_shampoo:
	python -m tdlgm.main --verbose --baseline lstm --shampoo_code true
	

baseline_tune:
	python -m tdlgm.main --verbose --baseline lstm --tune --reduced_dataset 0.001

tune:
	python -m tdlgm.main --verbose --tune --reduced_dataset 0.001

main:
	python -m tdlgm.main --verbose --reduced_dataset 0.001

shampoo:
	python -m tdlgm.main --shampoo_code true --verbose --epochs 100 

tDLGM:
	python -m tdlgm.tDLGM --verbose 

util:
	python -m tdlgm.util --verbose 

lint:
	./lint.sh
