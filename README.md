# surrogate-vs-implicit
Experiments comparing surrogate and implicit formulations for chemical process
models

# Copyright
The copyright header in `header.txt` must be added to every source (Python)
file in this repository.

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
