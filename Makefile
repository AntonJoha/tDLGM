.PHONY: main tDLGM util eval lint shampoo

_: main tune


tune_old:
	python -m experiments.main --verbose --tune --artifact_dir artifacts/tune --horizon 10 --use_old

baseline:
	python -m experiments.main --verbose --baseline  --artifact_dir artifacts/baseline_tune --horizon 20 --epochs 1000 --learning_rate 0.0001
	

eval_baseline:
	python -m experiments.eval --verbose  --checkpoint_path artifacts/baseline_tune/checkpoint_epoch0020_20260810-155003.pt


baseline_tune:
	python -m experiments.main --verbose --baseline --tune --artifact_dir artifacts/baseline_tune --horizon 10

tune:
	python -m experiments.main --verbose --tune --horizon 10

main:
	python -m experiments.main --verbose --learning_rate 0.0001 --horizon 20 --epochs 1000


eval:
	python -m experiments.eval --verbose  --checkpoint_path artifacts/tdlgm/checkpoint_epochbest_20260817-102926.pt

lint:
	./lint.sh
