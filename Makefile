.PHONY: main tDLGM util eval lint shampoo

_: main tDLGM util shampoo baseline baseline_shampoo tune


benchmark:
	python -m experiments.eval --verbose 

baseline:
	python -m experiments.main --verbose --baseline --reduced_dataset 0.001 --artifact_dir artifacts/baseline_tune --horizon 100 --epochs 10
	

eval_baseline:
	python -m experiments.eval --verbose  --checkpoint_path artifacts/baseline_tune/checkpoint_epochfinal_20260807-131328.pt

baseline_shampoo_tune:
	python -m experiments.main --verbose --baseline --shampoo_code --artifact_dir artifacts/baseline_tune --tune
	

baseline_shampoo:
	python -m experiments.main --verbose --baseline --shampoo_code --artifact_dir artifacts/baseline_tune 
	

baseline_tune:
	python -m experiments.main --verbose --baseline --tune --reduced_dataset 0.002 --artifact_dir artifacts/baseline_tune

tune:
	python -m experiments.main --verbose --tune --reduced_dataset 0.001 --horizon 100

main:
	python -m experiments.main --verbose --reduced_dataset 0.005 --learning_rate 0.0001 --horizon 100 --epochs 1000

rvae_main:
	python -m experiments.main_rvae --verbose --reduced_dataset 0.001 --learning_rate 0.01 

shampoo:
	python -m experiments.main --shampoo_code --verbose --epochs 100 


shampoo_tune:
	python -m tdlgm.main --shampoo_code --verbose --epochs 100 --tune



tDLGM:
	python -m tdlgm.tDLGM --verbose 

util:
	python -m experiments.util --verbose 

eval:
	python -m experiments.eval --verbose  --checkpoint_path artifacts/tdlgm/checkpoint_epochfinal_20260807-140956.pt

lint:
	./lint.sh
