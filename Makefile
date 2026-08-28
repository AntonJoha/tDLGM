.PHONY: main tDLGM util eval lint shampoo

_: main tune


tune_old:
	python -m experiments.main --verbose --tune --artifact_dir artifacts/tune --horizon 20 --use_old

baseline:
	python -m experiments.main --verbose --baseline  --artifact_dir artifacts/baseline_tune --horizon 20 --epochs 1000 --learning_rate 0.0001
	

eval_baseline:
	python -m experiments.eval --verbose  --checkpoint_path artifacts/baseline_tune/checkpoint_epochbest_20260827-092353.pt


baseline_tune:
	python -m experiments.main --verbose --baseline --tune --artifact_dir artifacts/baseline_tune --horizon 20

tune:
	python -m experiments.main --verbose --tune --horizon 20

main:
	python -m experiments.main --verbose --learning_rate 0.0001 --horizon 20 --epochs 1


eval_new:
	python -m experiments.eval --verbose  --checkpoint_path artifacts/tdlgm/checkpoint_epochbest_20260827-101909.pt

eval:
	python -m experiments.eval --verbose  --checkpoint_path artifacts/tune/checkpoint_epochbest_20260827-092541.pt

eval_debug:
	python -m experiments.eval --verbose  --checkpoint_path artifacts/tdlgm/checkpoint_epochfinal_20260828-103115.pt

lint:
	./lint.sh
