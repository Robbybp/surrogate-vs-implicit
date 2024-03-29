# Surrogate models vs. implicit functions
Comparing the two approaches for optimization of process flowsheets involving
challenging-to-converge unit models.

# Installation
This repository is structured as a small Python package to facilitate code
organization and testing. It can be installed with:
```bash
$ python setup.py develop
```
Then functionality can be imported in Python from the `svi` package:
```python
from svi.distill import create_instance
model = create_instance()
```

# Repository structure
This work started as exploratory research code, and has morphed into its present
state over time. Only the code in the `svi/auto_thermal_reformer` directory
should be expected to work. This is the code used to produce the results of the
FOCAPD paper cited below.

# Citation
If you use this code in your research, please cite the following:
```bibtex
@inproceedings{bugosen2024focapd,
author={Bugosen, Sergio I. and Laird, Carl D. and Parker, Robert B.},
title={Chemical process flowsheet optimization with full space, surrogate, and implicit formulations of a {G}ibbs reactor},
booktitle={Foundations of Computer-Aided Process Design},
note={Accepted},
year={2024},
}
```
A preprint may be found at [https://arxiv.org/abs/2310.09307](https://arxiv.org/abs/2310.09307).

# Copyright
The copyright header in `header.txt` must be added to every source (Python)
file in this repository.
