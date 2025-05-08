# Vocal communication is seasonal in social groups of wild, free-living house mice

This repository contains code needed to reproduce figures and analyses from "Vocal communication is seasonal in social groups of wild, free-living house mice" (Jourjine et al. 2025 Proceedings B). It contains two directories.

### 1. notebooks
- Contains Jupyter notebooks for performing analyses and creating figures

### 2. `src`
- Contains helper functions used within the Jupyter notebooks.

## How to use

The code in this repository is intended to be used along with data at the Dryad repository here: ###

To do this, take the following steps:
1. Clone or download this repository to your local machine. You should get a folder called wild-mus-vocal-ecology.
2. Download the data folder at ###, then unzip it by clicking on it, running `tar -xf path/to/wild-mus-vocal-ecology-data.zip` (Windows PowerShell) or `unzip path/to/wild-mus-vocal-ecology-data.zip` (MacOS terminal). You should get a folder called wild-mus-vocal-ecology-data containing four directories: "data", "models", "parameters", and "annotations"
3. Copy or move the contents of the data folder to the folder you cloned or downloaded from here (these should be the folders "data", "models", "parameters", and "annotations").\
To copy from the command line:  
`rsync -ahP /path/to/wild-mus-vocal-ecology-data/ /path/to/wild-mus-vocal-ecology/`\
To move from the command line:
`mv path/to/wild-mus-vocal-ecology-data/* /path/to/wild-mus-vocal-ecology/`

4. Set up the necessary virtual environments and access the analysis notebooks using the steps below:

Download and install Anaconda following the instructions here if you haven't already done so: 

`https://docs.anaconda.com/getting-started/`

Then run the following in the command line:


	conda env create -f audiomoth_environment.yml -n audiomoth -v 
	conda env create -f das_environment.yml -n das -v 

	
In the root directory of your project (where the src directory resides), install the necessary helper functions and set up Jupyter kernels by running:

	conda activate audiomoth
	python -m ipykernel install --user --name audiomoth --display-name "audiomoth"
	pip install -e .
	conda deactivate
	conda activate das
	pip install -e .
	python -m ipykernel install --user --name das --display-name "DAS"
	conda deactivate
	
This ensures that the helper functions are accessible in the notebooks and creates dedicated Jupyter kernels for each environment, allowing you to switch between them within a single notebook.

Then run

	conda activate audiomoth
	jupyter notebook
	
from the root directory to launch jupyter, navigate to the notebooks directory, and select the notebook you would like to use.


