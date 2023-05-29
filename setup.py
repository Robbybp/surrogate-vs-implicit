from setuptools import setup

setup(
    name="surrogate-vs-implicit",
    description=(
        "Experiments comparing surrogate and implicit function formulations"
        " for chemical process models",
    ),
    author="Robert Parker and Sergio Bugosen",
    license="BSD-3",
    packages=["svi"],
)
