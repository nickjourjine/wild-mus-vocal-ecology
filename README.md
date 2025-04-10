# Vocal communication is seasonal in social groups of wild, free-living house mice

This dataset contains code and data required to reproduce figures and analyses from XXX (reference removed for double blind peer review). It is organized into six directories:

### 1. notebooks
- Contains Jupyter notebooks for performing analyses and creating figures

### 2. `src`
- Contains helper functions used within the Jupyter notebooks.

### 3. data
- Comprises four subdirectories, each holding raw data for analyses:

  #### phenotypes
  - Holds the following CSV files that describe barn population checks. 
  
   `sexes.csv` is a table mapping transponder IDs for each mouse to its sex, as confirmed by hand when the mouse was first transpondered on the basis of genital morphology. All transponder IDs have been anonymized.
    - Columns are (`transponder_id`), the transponder ID of the mouse the first time it was caught and sex (`Sex`), its sex as determined by an expert at the time of transpondering on the basis of genital morphology.

    `All_popul_checks_update2023_transponder-and-popchecks.csv` records the size of the barn population at each population check between 2003 and 2023
    - Columns are the date of the check (`Date`) and the total number of mice caught (`Total mice`).
    
    `Pups_2006_to_2023.csv` describes the pups found in the barn between 2006 and 2023. 
    - Columns are the date of the check (`date` and `Datetext`) and the total number of pups found on that date (`total_littersize`).

  #### rfid
  - Data from the barn antenna system.
  - Subdirectories:
    - **box_events**
      - Data tables (.feather format) with one row for each box event (entrance or exit)
      - Analyzed columns are event time (`event_time`), number of mice post-event (`num_partners_after_event`), box location (`box`), and event type (`event_type`: 1 for entrance, 2 for exit). Other columns are internal database IDs used to detect accidental row duplication.
      - File naming convention: Files start with two dates in `yyyymmdd` format indicating the time range of the data.
    - **mouse_meets**
      - Data tables (.feather format) with one row for each meeting between unique pairs of mice
      - Analyzed columns are start (`overlap_start_time`) and end time (`overlap_end_time`), duration (`time_in_secs`), mouse IDs (`id1`, `id2`), and box location (`box`). Other columns are internal database IDs used to detect accidental row duplication.
      - File naming convention: Files start with two dates in `yyyymmdd` format indicating the time range of the data.
    - **mouse_stays**
      - Data tables (.feather format) with one row for each stay by a mouse in a box
      - Analyzed columns are transponder ID (`transponder_id`), box location (`box`), duration (`time_in_secs`), start (`entry_time`) and end timestamps (`exit_time`). Other columns are internal database IDs used to detect accidental row duplication.
      - File naming convention: Files start with two dates in `yyyymmdd` format indicating the time range of the data.

  #### segments
  - Data about detected vocalizations from recorded boxes
  - Subdirectories:
    - **vocal_counts**
      - Data tables (.csv format) with one row for each consecutive recorded 55 second interval, for each audiomoth.
      - Columns include the timestamp of the start of the interval in audiomoth format (`minute`), the timestamp in yyyy-mm-dd hh:mm:ss format (`audiomoth_timestamp`), the number of USVs detected in that interval (`USV_count`), the number of squeaks detected in that interval (`squeak_count`), whether the sun was up during the interval (`sunup`), the dates of the audiomoth deployment in which these vocalizations were detected (`deployment`), the audiomoth that recorded the vocalizations (`audiomoth`), and the box they were recorded from (`box`)
      - File naming convention: yyyymmdd-yyyymmdd_box#_counts where yyyymmdd-yyyymmdd is the date range of the depoloyment and # is the box number recoded during that deployment to which the data file corresponds. 
    - **vocal_events**
      - Data tables (.csv format) with one row for each detected vocalization
      - Columns include the deployment date when the vocalization was detected (`deployment`), the audiomoth that recorded it (`audiomoth`), the box it was recorded in (`box`), the timestamp of the wav file it was detected in (`audiomoth_timestamp`), the start and end of the vocalization relative to the beginnning of that wav file (`start_seconds` and `stop_seconds`, respectively), its duration (`duration`), the path to the model that assigned that label (`model`), the original location of the wav file containing the vocalization when inference was performed (`source_file`), the human-interpretable label of the vocalization (`label`), the value in the audiomoth_timestamp column in datetime format (`audiomoth_timestamp_datetime`), the absolute start and stop time of the vocalization using the audiomoth internal clock (`audiomoth_start_seconds` and `audiomoth_stop_seconds`). Note that when inference was performed the "squeak" vocalizations were assigned the label "cry". All "cry" labels were replaced with "squeak" prior to analyses for consistency.
      - File naming convention: yyyymmdd-yyyymmdd_box#_segments or yyyymmdd-yyyymmdd_box#_time-adjusted where yyyymmdd-yyyymmdd is the date range of the depoloyment and # is the box number recoded during that deployment to which the data file corresponds. Files ending in 'time-adjusted' contain the following additional columns with timestamps adjusted so that they are aligned to the RFID system (see methods for details): 
        - `audiomoth_start_seconds_adjusted`: start of the vocalization adjusted to match the RFID system clock
        - `audiomoth_stop_seconds_adjusted`: end of the vocalization adjusted to match the RFID system clock
        - `audiomoth_timestamp_datetime_adjusted`: start of the audiomoth wav file adjusted to match the RFID system clock
        - `deployment_correction_seconds`: time difference in seconds between RFID and audiomoth clock at start of the deployment
        - `recovery_correction_seconds`: time difference in seconds between RFID and audiomoth clock at end of the deployment
        - `estimated_or_actual_time_correction`: whether the rate of clock drift was calculated directly from this audiomoth during this deployment ('actual') or from an average (`estimated`) rate of clock drift based on deployments when this could be directly measured from this audiomoth (ie when an acoustic chime was used at both the start and end of the recording).

  #### umap
  - UMAP coordinates for vocalizations plotted in Figure 3, Panel A. See the notebook `Figure 4.ipynb` for details.
  - The spectrograms directory contains example spectrograms from Figure 4, panel B

### models
- Contains artifacts related to the DAS model, generated following training using the `DAS` package:
  - Final trained model: `20230219_120047_model.h5`
  - Training parameters: `20230219_120047_params.yaml`
  - Evaluation file: `20230219_120047_results.h5`

### parameters
- Includes:
  - Images of the barn
  - Floor plan coordinates
  - Colormaps for figures

### annotations
- Contains hand-annotated vocalizations and their acoustic features. Used for model evaluation as shown in Supplemental Figure 3. See the notebook `Supplemental Figure 3.ipynb` for details.

## Code/Software

The dataset contains two .yml files with packages required to perform analyses, one for anlayses involving DAS (das_environment.yml) and one for all other analyses (audiomoth_environment.yml). To create conda envrionments from these files, first download and install Anaconda following the instructions here: 

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


