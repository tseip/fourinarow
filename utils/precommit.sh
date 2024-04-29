#!/bin/bash

# Dependencies for running the precommit formatting script include:
# clang (clang-format specifically)
# autopep8 (pip install autopep8)
# doxygen

# Run clang format on cpp files
./run-clang-format.py  --style Google -i ../*.cpp
./run-clang-format.py  --style Google -i ../*.h

# Run autopep8 on python files
autopep8 ../model_fitting/*.py -i

pushd ..
doxygen Doxyfile
