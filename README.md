# Protein Folding Energy

Compile datasets of protein folding energies.


## Protherm

Parse the raw text file from the [Protherm website](http://www.abren.net/protherm/).

This is actually a lot of hassle, see: `notebooks/protherm.ipynb`.

Validate the results against **Protherm*** and exclude mutations (~86) that are present in both but have different values (most of these are the same value with different sign :worried:).


## Protherm*

Parse CSV files from the [Rosetta ΔΔG benchmark](https://github.com/Kortemme-Lab/ddg) repository.


