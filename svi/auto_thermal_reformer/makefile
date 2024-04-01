SUBSET = "0,2,4,6,7,8,14,18,21,24,25,26,27,30,32,35,36,37,38,39,41,42,44,45,47"

all: analysis, plots

plots:
	python plot_convergence.py \
	    data/fullspace-sweep-cyipopt.csv \
	    --feastol=1e-5 \
	    --no-legend \
	    --title="Full-space"
	python plot_convergence.py \
	    data/alamo-sweep-cyipopt.csv \
	    --validation-fpath=data/alamo-sweep-cyipopt-validation.csv \
	    --feastol=1e-5 \
	    --no-legend \
	    --title="ALAMO surrgate"
	python plot_convergence.py \
	    data/nn-sweep-full.csv \
	    --validation-fpath=data/nn-sweep-full-validation.csv \
	    --feastol=1e-5 \
	    --no-legend \
	    --title="Neural network surrgate"
	python plot_convergence.py \
	    data/implicit-sweep.csv \
	    --validation-fpath=data/implicit-sweep-validation.csv \
	    --feastol=1e-5 \
	    --title="Implicit function"

analysis:
	python analyze_results.py \
	    data/fullspace-sweep-cyipopt.csv \
	    --feastol=1e-5\
		--subset=$(SUBSET)
	python analyze_results.py \
	    data/alamo-sweep-cyipopt.csv \
	    --validation-fpath=data/alamo-sweep-cyipopt-validation.csv \
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
