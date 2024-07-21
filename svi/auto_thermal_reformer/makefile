SUBSET = "0,2,4,6,7,8,14,18,21,24,25,26,27,30,32,35,36,37,38,39,41,42,44,45,47"

train:
	python generate_training_data.py --regular-samples
	python generate_alamo_surrogate.py data/training-data.csv
	python nn_tuning_training.py data/training-data.csv --tune

sweep:
	python run_fullspace_sweep.py
	python run_alamo_sweep.py
	python run_nn_sweep.py
	python run_implicit_sweep.py

validate:
	python validate_sweep_results.py data/alamo-sweep.csv --baseline-fpath=data/fullspace-sweep.csv
	python validate_sweep_results.py data/nn-sweep-full.csv --baseline-fpath=data/fullspace-sweep.csv
	python validate_sweep_results.py data/implicit-sweep.csv --baseline-fpath=data/fullspace-sweep.csv

analyze:
	python analyze_results.py \
	    data/fullspace-sweep.csv \
	    --feastol=1e-5\
		--subset=$(SUBSET)
	python analyze_results.py \
	    data/alamo-sweep.csv \
		--validation-fpath=data/alamo-sweep-validation.csv \
	    --feastol=1e-5 \
		--subset=$(SUBSET)
	python analyze_results.py \
	    data/nn-sweep-full.csv \
		--validation-fpath=data/nn-sweep-full-validation.csv \
	    --feastol=1e-5 \
		--subset=$(SUBSET)
	python analyze_results.py \
	    data/implicit-sweep.csv \
		--validation-fpath=data/implicit-sweep-validation.csv \
	    --feastol=1e-5 \
		--subset=$(SUBSET)

plot:
	python plot_convergence.py \
	    data/fullspace-sweep.csv \
	    --feastol=1e-5 \
	    --no-legend \
	    --title="Full-space"
	python plot_convergence.py \
	    data/alamo-sweep.csv \
	    --validation-fpath=data/alamo-sweep-validation.csv \
	    --feastol=1e-5 \
	    --no-legend \
	    --title="ALAMO surrgate" \
		--show-training-bounds
	python plot_convergence.py \
	    data/nn-sweep-full.csv \
	    --validation-fpath=data/nn-sweep-full-validation.csv \
	    --feastol=1e-5 \
	    --no-legend \
	    --title="Neural network surrgate" \
		--show-training-bounds
	python plot_convergence.py \
	    data/implicit-sweep.csv \
	    --validation-fpath=data/implicit-sweep-validation.csv \
	    --feastol=1e-5 \
	    --title="Implicit function"

plot-error:
	python plot_surrogate_error.py \
		data/alamo-sweep-surrogate-error.csv \
		--no-legend \
		--opaque \
		--title="ALAMO"
	python plot_surrogate_error.py \
		data/nn-sweep-full-surrogate-error.csv \
		--opaque \
		--title="Neural network"

plot-error-png:
	python plot_surrogate_error.py \
		data/alamo-sweep-surrogate-error.csv \
		--no-legend \
		--opaque \
		--title="ALAMO" \
		--plot-fname=alamo-sweep-surrogate-error.png
	python plot_surrogate_error.py \
		data/nn-sweep-full-surrogate-error.csv \
		--opaque \
		--title="Neural network" \
		--plot-fname=nn-sweep-full-surrogate-error.png
