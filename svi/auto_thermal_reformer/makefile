all:
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
