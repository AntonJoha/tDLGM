.PHONY: main tDLGM util eval lint shampoo

_: main tDLGM util shampoo baseline baseline_shampoo tune


baseline:
	python -m tdlgm.main --verbose --baseline --reduced_dataset 0.001
	

baseline_shampoo:
	python -m tdlgm.main --verbose --baseline --shampoo_code
	

baseline_tune:
	python -m tdlgm.main --verbose --baseline --tune --reduced_dataset 0.001

tune:
	python -m tdlgm.main --verbose --tune --reduced_dataset 0.001

main:
	python -m tdlgm.main --verbose --reduced_dataset 0.001

shampoo:
	python -m tdlgm.main --shampoo_code --verbose --epochs 100 

tDLGM:
	python -m tdlgm.tDLGM --verbose 

util:
	python -m tdlgm.util --verbose 

eval:
	python -m tdlgm.eval --verbose 

lint:
	./lint.sh
