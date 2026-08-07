.PHONY: main tDLGM util eval lint shampoo

_: main tDLGM util shampoo baseline baseline_shampoo tune


benchmark:
	python -m tdlgm.eval --verbose 

baseline:
	python -m tdlgm.main --verbose --baseline --reduced_dataset 0.001 --artifact_dir artifacts_dev/baseline_tune
	


baseline_shampoo_tune:
	python -m tdlgm.main --verbose --baseline --shampoo_code --artifact_dir artifacts/baseline_tune --tune
	

baseline_shampoo:
	python -m tdlgm.main --verbose --baseline --shampoo_code --artifact_dir artifacts/baseline_tune 
	

baseline_tune:
	python -m tdlgm.main --verbose --baseline --tune --reduced_dataset 0.002 --artifact_dir artifacts/baseline_tune

tune:
	python -m tdlgm.main --verbose --tune --reduced_dataset 0.002

main:
	python -m tdlgm.main --verbose --reduced_dataset 0.005 --learning_rate 0.0001

rvae_main:
	python -m tdlgm.main_rvae --verbose --reduced_dataset 0.001 --learning_rate 0.01 

shampoo:
	python -m tdlgm.main --shampoo_code --verbose --epochs 100 


shampoo_tune:
	python -m tdlgm.main --shampoo_code --verbose --epochs 100 --tune



tDLGM:
	python -m tdlgm.tDLGM --verbose 

util:
	python -m tdlgm.util --verbose 

eval:
	python -m tdlgm.eval --verbose 

lint:
	./lint.sh
