# Vocal communication is seasonal in social groups of wild, free-living house mice

This repository contains code needed to reproduce figures and analyses from "Vocal communication is seasonal in social groups of wild, free-living house mice" (Jourjine et al. 2025). It contains two directories.

### 1. `notebooks`
- Contains Jupyter notebooks for performing analyses (one per figure).

### 2. `src`
- Contains helper functions used within the Jupyter notebooks.

## How to use

The code in this repository is intended to be used along with data at the Dryad repository [here](link).

To combine code and data:
1. Clone or download this repository to your local machine by clicking the big green `<> Code` button above. You should get a folder called wild-mus-vocal-ecology.
2. Download the data folder [here](link), then unzip it by clicking on it, or running
    - `unzip path/to/wild-mus-vocal-ecology-data.zip` (MacOS Terminal)  
    - `Expand-Archive -Path path\to\wild-mus-vocal-ecology-data.zip -DestinationPath path\to\output-folder` (Windows Powershell) 
	
	
	You should end up with a folder called wild-mus-vocal-ecology-data containing four directories: "data", "models", "parameters", and "annotations".  
	
3. Copy or move the contents of the wild-mus-vocal-ecology-data folder (not the folder itself) to the wild-mus-vocal-ecology folder you cloned or downloaded from this repository.  

    - To copy:  
	    MacOS Terminal:  
    	`rsync -ahP /path/to/wild-mus-vocal-ecology-data/ /path/to/wild-mus-vocal-ecology/`  
		Windows Powershell:  
	    `Copy-Item -Path "C:\path\to\wild-mus-vocal-ecology-data\*" -Destination "C:\path\to\wild-mus-vocal-ecology" -Recurse` 

    - To move:  
	    MacOS Terminal:  
        `rsync -ahP --remove-source-files /path/to/wild-mus-vocal-ecology-data/ /path/to/wild-mus-vocal-ecology/`  
		Windows Powershell:  
	    `Move-Item -Path "C:\path\to\wild-mus-vocal-ecology-data\*" -Destination "C:\path\to\wild-mus-vocal-ecology"` 

4. Set up the necessary virtual environments and access the analysis notebooks using the steps below:

Download and install Anaconda following the instructions here if you haven't already done so: 

`https://docs.anaconda.com/getting-started/`

Then run the following in your terminal (Powershell on Windows, Terminal app on Mac/Linux) to create the virtual environments:


	conda env create -f audiomoth_environment.yml -n audiomoth -v 
	conda env create -f das_environment.yml -n das -v 

	
Move to the wild-mus-vocal-ecology directory:
	
   Mac/Linux: `cd path/to/wild-mus-vocal-ecology`  
   Windows Powershell: `cd C:\path\to\wild-mus-vocal-ecology` 

Then install the necessary helper functions and set up Jupyter kernels by running:

	conda activate audiomoth
	python -m ipykernel install --user --name audiomoth --display-name "audiomoth"
	pip install -e .
	conda deactivate
	conda activate das
	pip install -e .
	python -m ipykernel install --user --name das --display-name "DAS"
	conda deactivate
	
This ensures that the helper functions are accessible in the notebooks and creates dedicated Jupyter kernels for each environment, allowing you to switch between them within a single notebook.

Then run the following 

	conda activate audiomoth
	jupyter notebook
	
to launch Jupyter (a browser window should open - if it doesn't, you can copy/paste the link that appears in the terminal window following these commands), navigate to the notebooks directory, and select the notebook you would like to use.

If you have trouble completing any of this steps, please let me know by raising an issue! Just click the `Issues` button at the top of the page, then the big green `New Issue` button.


