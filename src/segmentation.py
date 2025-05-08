# this file contains functions for using Deep Audio Segmenter to find vocalizations

import glob
import os
import json

import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.interpolate import interp2d
from scipy.signal import stft
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import shutil	
import seaborn as sns 
from scipy.ndimage.filters import gaussian_filter
from datetime import date, datetime, timedelta
from joblib import Parallel, delayed

from src.timestamps import check_sunup, make_time_columns
from src.filespaths import combine_dataframes
from src.parameters import load_json


def das_predict(wav, model, params, verbose, segment_minlen, segment_fillgap, segment_thres, pad = True):
    """
    Segment a wav file with a das model. Simple wrapper for das.predict.predict
    
    Parameters:
        model_name (string): the full path to the trained das model
        audio_dir (string): the path to the directory containing the raw wav files to be segmented (ie one deployment of one moth)
        save_dir (string): the path to the directory to save the predicted start and stop times
        segment_minlen: the minimum length of a segment in seconds. Segments below this length will be ignored
        segment_fillgap: the minimum inter-segment interval below which two segments will be merged into one
        segment_thres: confidence threshold for detecting segments. Range 0..1. Defaults to 0.5.

    Returns:
        None
    
    
    """
    #import das
    import das.predict
	
    #prepare the wav
    samplerate, x = wavfile.read(wav)
    x = np.atleast_2d(x).T

    #predict
    events, segments, class_probabilities, class_names = das.predict.predict(x=x, 
                                                                             model=model, 
                                                                             params=params,
                                                                             verbose=verbose,
                                                                             segment_minlen=segment_minlen,
                                                                             segment_fillgap=segment_fillgap, 
                                                                             segment_thres=segment_thres,
                                                                             pad=True)
    
    return events, segments, class_probabilities, class_names

    
def segments_to_dataframe(segments, deployment, segment_minlen, segment_fillgap, segment_thres, model_name, save=False, wav=None, save_dir=None, new_das=False):
    """
    Make a dataframe of das predictions from the segments dictionary returned by das predict

    Parameters:
        segments (dict): output of das_predict
        save_dir (str): directory to save the csv
        deployment (str): deployment in yyyymmdd-yyyymmdd format
        wav (str): full path to the wav file that this csv is about with file extension
        segment_minlen: the minimum length of a segment in seconds. Segments below this length will be ignored
        segment_fillgap: the minimum inter-segment interval below which two segments will be merged into one
        segment_thres: confidence threshold for detecting segments. Range 0..1. Defaults to 0.5.
        model_name: path to the model used for prediction
        save (bool): if True, save a csv of the dataframe

    Returns:
        None
    """

    df = pd.DataFrame()
    df['deployment'] = [deployment.split('_')[0]] * len(segments['onsets_seconds'])
    df['moth'] = [wav.split('/')[-3]] * len(segments['onsets_seconds'])
    df['box'] = [deployment.split('_')[-1]] * len(segments['onsets_seconds'])
    df['minute'] = [wav.split('/')[-1].split('.')[0]] * len(segments['onsets_seconds'])
    df['start_seconds'] = segments['onsets_seconds']
    df['stop_seconds'] = segments['offsets_seconds']
    
    if not new_das:
        df['duration'] = segments['durations_seconds']
    else:
        df['duration'] = df['stop_seconds'] - df['start_seconds']
    
    df['label'] = segments['sequence']
    df['model'] = model_name
    df['segment_threshold'] = segment_thres
    df['segment_min_len'] = segment_minlen
    df['segment_fillgap'] = segment_fillgap

    if save:
        df.to_csv(os.path.join(save_dir, f"{deployment}_{wav.split('/')[-1].split('.')[0]}.csv"), index=False)

    return df
    
    if save:
        df.to_csv(os.path.join(save_dir, deployment+'_'+wav.split('/')[-1].split('.')[0]+'.csv'), index=False)
        
    return df
    
def get_segments(model_name, model, params, wav, save_dir, segment_minlen, segment_fillgap, segment_thres ):
    """
    Take a wav file, segment it, and save the predictions to a csv

    Parameters:
        model : the trained das model
        params : model parameters
        wav (string): the path to the wav
        save_dir (string): the path to the directory to save the predicted start and stop times
        segment_minlen: the minimum length of a segment in seconds. Segments below this length will be ignored
        segment_fillgap: the minimum inter-segment interval below which two segments will be merged into one
        segment_thres: confidence threshold for detecting segments. Range 0..1. Defaults to 0.5.


    Returns:
        None
    """

    import das
    import das.utils
    import das.predict
    from tqdm import tqdm

    from src import timestamps


    #get wav file paths for das predict
    #deployment = wav.split('/')[-2].split('_box')[0]
    deployment = wav.split('/')[-2].split('_')[0]
    

    #predict
    _, segments, _, _ = das_predict(wav=wav, 
                                    model=model, 
                                    params=params,
                                    verbose=2,
                                    segment_minlen=segment_minlen,
                                    segment_fillgap=segment_fillgap, 
                                    segment_thres=segment_thres,
                                    pad=True)

   
    #if there are vocalizations, write to csv
    if len(segments['onsets_seconds']) != 0:
        save_das_predict(segments=segments, 
                         save_dir=save_dir, 
                         deployment=deployment, 
                         wav=wav, 
                         segment_thres=segment_thres, 
                         segment_minlen=segment_minlen, 
                         segment_fillgap=segment_fillgap, 
                         model_name=model_name)

    else:
        return


    
    
def get_segments_batch(model_name, audio_dir, save_dir, segment_minlen, segment_fillgap, segment_thres):

    """
    Segment all files in a deployment using a das model. This is a backup function for troubleshooting. 

    Parameters:
        model_name (string): the full path to the trained das model
        audio_dir (string): the path to the directory containing the raw wav files to be segmented (ie one deployment of one moth)
        save_dir (string): the path to the directory to save the predicted start and stop times
        segment_minlen: the minimum length of a segment in seconds. Segments below this length will be ignored
        segment_fillgap: the minimum inter-segment interval below which two segments will be merged into one
        segment_thres: confidence threshold for detecting segments. Range 0..1. Defaults to 0.5.

    Returns:
        None

    """

    import das
    import das.utils
    import das.predict
    from tqdm import tqdm
    
    from src import timestamps

    #load the model and parameters
    model, params = das.utils.load_model_and_params(model_name)
    
    #get wav file paths for das predict
    deployment = os.path.split(audio_dir)[-1]
    
    #get the wav files
    audio_for_predict = [os.path.join(audio_dir,i) for i in os.listdir(audio_dir) if not i.startswith ('.') and i.endswith('wav')]
    
    #exclude wav files that are zero bytes (written after SD card is full)
    audio_for_predict = [i for i in audio_for_predict if os.path.getsize(i) != 0]
    
    #exclude wav files you already have predictions for
    if len([i for i in os.listdir(save_dir) if i.endswith('.csv')]) > 0:
        all_timestamps = [('_').join(i.split('_')[-2:]).split('.')[0] for i in os.listdir(save_dir) if i.endswith('.csv')]
        last_timestamp = max([timestamps.audiomoth_to_datetime(i) for i in all_timestamps])
        audio_for_predict = [i for i in audio_for_predict if timestamps.audiomoth_to_datetime(i.split('/')[-1].split('.')[0]) > last_timestamp]
    
    #exclude first and last 10 minutes
    audio_for_predict = sorted(audio_for_predict)[10:-10]
    
    #initalize list to collect predictions
    all_predictions = []
    
    for wav in tqdm(audio_for_predict):

        #check if you already have segmented 
        if deployment+'_'+wav.split('/')[-1].split('.')[0]+'.csv' in os.listdir(save_dir):
            pass

        else: #segment if not

            #get wav file paths for das predict
            deployment = wav.split('/')[-2].split('_box')[0]

            #predict
            _, segments, _, _ = das_predict(wav=wav, 
                                            model=model, 
                                            params=params,
                                            verbose=2,
                                            segment_minlen=segment_minlen,
                                            segment_fillgap=segment_fillgap, 
                                            segment_thres=segment_thres,
                                            pad=True)


            #if there are vocalizations, write to csv
            if len(segments['onsets_seconds']) != 0:
                segments_to_dataframe(segments=segments, 
                                      save_dir=save_dir, 
                                      deployment=deployment, 
                                      wav=wav, 
                                      segment_thres=segment_thres, 
                                      segment_minlen=segment_minlen, 
                                      segment_fillgap=segment_fillgap,
                                      model_name=model_name, 
                                      save=True)

    print('done.')


    
    
def get_segments_from_list(model_name, deployment, audio_dir, audio_list, save_dir, segment_minlen, segment_fillgap, segment_thres, parallel, n_jobs):
    """
    Segment all files in a deployment using a das model. This is a backup function for troubleshooting. 
    Same as get_voc_segments but provide a list instead
    Useful for predicting just a subset of a recorindg.

    Parameters:
        model_name (string): the full path to the trained das model
        audio_dir (string): the path to the directory containing the raw wav files to be segmented (ie one deployment of one moth)
        audio_list (list): list of paths ro audio to predict that are in audio_dir
        save_dir (string): the path to the directory to save the predicted start and stop times
        segment_minlen: the minimum length of a segment in seconds. Segments below this length will be ignored
        segment_fillgap: the minimum inter-segment interval below which two segments will be merged into one
        segment_thres: confidence threshold for detecting segments. Range 0..1. Defaults to 0.5.
        parallel (bool): if true use joblib
        n_jobs (int): number of jobs parllel jobs for joblib
        

    Returns:
        None

    """

    import das
    import das.utils
    import das.predict
    from tqdm import tqdm
    
    from src import timestamps

    #load the model and parameters
    model, params = das.utils.load_model_and_params(model_name)
    
    #get wav file paths for das predict
    #deployment = os.path.split(audio_dir)[-1]
    
    #get the wav files
    audio_for_predict = audio_list
    
    #exclude wav files that are zero bytes (written after SD card is full)
    audio_for_predict = [i for i in audio_for_predict if os.path.getsize(i) != 0]
    
    #initalize list to collect predictions
    all_predictions = []
    
    for wav in tqdm(audio_for_predict):

        #predict
        _, segments, _, _ = das_predict(wav=wav, 
                                        model=model, 
                                        params=params,
                                        verbose=2,
                                        segment_minlen=segment_minlen,
                                        segment_fillgap=segment_fillgap, 
                                        segment_thres=segment_thres,
                                        pad=True)


        #if there are vocalizations, write to csv
        if len(segments['onsets_seconds']) != 0:
            segments_to_dataframe(segments=segments, 
                                  save_dir=save_dir, 
                                  deployment=deployment, 
                                  wav=wav, 
                                  segment_thres=segment_thres, 
                                  segment_minlen=segment_minlen, 
                                  segment_fillgap=segment_fillgap,
                                  model_name=model_name, 
                                  save=True)

    print('done.')
    
def collect_segments(save_directory, model, root, moths):
    """
    Collect the segment csv files from get_segments() into a single csv per deployment and save 

    Parameters:
        
        save_directory: directory containing the vocalization counts by deplyment, with subdirectories for each audiomoth
        model: the name of the model (date and time ID in format yyyymmdd_hhmmss) that generated the segments
		root: the root directory of the project folder
       
    Returns:
        None

    """
    source_dir = os.path.join(root, 'data', 'segments', 'raw', model) # location of the raw predictions to aggregate
    box_dict = load_json(os.path.join(root, 'parameters', 'boxes_recorded.json'))
    locations_dict = load_json(os.path.join(root, 'parameters', 'recording_storage_locations.json'))
    deployment_dates = load_json(os.path.join(root,'parameters', 'deployment_dates.json'))
    label_names = load_json(os.path.join(root, 'parameters', 'vocalization_label_names.json'))
    
    print('collecting vocal events...')
    for moth in moths: 
        
        print(moth)

        deployments = os.listdir(os.path.join(source_dir, moth))

        for deployment in deployments:
			
            print('\t', deployment)
            box = box_dict[moth][deployment]
            box_string = ('').join(['box',str(box_dict[moth][deployment])])
            
            if ('_').join([deployment,box_string, 'segments.csv']) not in os.listdir(os.path.join(save_directory, model)):
                
				#combine into one big csv
                all_df = pd.concat([pd.read_csv(i) for i in glob.glob(os.path.join(source_dir, moth, deployment, deployment+'*.csv'))])
                
                #improve column names
                all_df = all_df.rename(columns={'label':'das_label', 'minute':'audiomoth_timestamp'})
                
                #add source file
                audio_root = os.path.join(locations_dict[deployment], moth, ('_').join([deployment, box_string]))
                all_df['source_file'] = [os.path.join(audio_root, i+'.wav') for i in all_df['audiomoth_timestamp']]
                
                #add box
                all_df['box'] = box
                
                #add interpretable labels
                all_df['label'] = all_df['das_label'].astype(str).map(label_names)
                
                #add start and stop timestamps
                all_df = make_time_columns(df=all_df, audiomoth_timestamp_column = 'audiomoth_timestamp')
                
                #save
                #all_df.to_csv(os.path.join(save_directory, model, 'by_audiomoth',moth, ('_').join([deployment,moth, 'segments.csv'])), index=False)
                all_df.to_csv(os.path.join(save_directory, model, ('_').join([deployment,box_string, 'segments.csv'])), index=False)
    
    print('done.')   
            
  
    
def get_counts(segments_df, root):
	"""
	take a df from a single deployment where each row is a vocalization (eg, output of collect_segments) 
	return a df where each row is a recorded minute and columns are USV and cry counts for that minute.
	"""

	from src.timestamps import audiomoth_to_datetime, check_sunup
	from src.parameters import load_json

	box_dict = load_json(os.path.join(root, 'parameters', 'boxes_recorded.json'))
	data_location_dict = load_json(os.path.join(root, 'parameters', 'recording_storage_locations.json'))

	deployment = list(segments_df['deployment'].unique())
	moth = list(segments_df['moth'].unique())
	assert len(deployment) == len(moth) == 1, "More than one audiomoth or deployment in this csv - something went wrong aggregating vocal events from raw)"
	deployment = deployment[0]
	moth = moth[0]

	# Get the data
	segments_df['audiomoth_timestamp'] = segments_df['audiomoth_timestamp'].astype(str)
	data_source = data_location_dict[deployment]
	box = box_dict[moth][deployment]

	# Get the times to iterate through - need to access the actual timestamps 
	audio_dir = os.path.join(data_source, moth, '_'.join([deployment, 'box' + str(box)]))
	minutes = sorted([i.split('.')[0] for i in os.listdir(audio_dir) if not i.startswith('.') and i.endswith('wav')])

	# Initialize DataFrame for counts
	count_df = pd.DataFrame(minutes, columns=['minute'])

	# Group by 'audiomoth_timestamp' and 'label', then count occurrences
	grouped = segments_df.groupby(['audiomoth_timestamp', 'label']).size().unstack(fill_value=0).reset_index()

	# Merge grouped counts with the count_df
	count_df = count_df.merge(grouped, left_on='minute', right_on='audiomoth_timestamp', how='left').fillna(0)

	#deal with recordings that don't have any of one type
	if not 'cry' in count_df.columns:
		count_df['cry'] = 0
	if not 'USV' in count_df.columns:
		count_df['USV'] = 0
	if not 'noise' in count_df.columns:
		count_df['noise'] = 0

	
	# Rename columns for clarity
	count_df = count_df.rename(columns={'cry': 'squeak_count', 'USV': 'USV_count', 'noise': 'noise_count'})
	
	for col in ['squeak_count', 'USV_count', 'noise_count']:
		count_df[col] = count_df[col].astype(int)

	# Convert minute column to datetime and check sunup
	count_df['audiomoth_timestamp'] = [audiomoth_to_datetime(minute) for minute in count_df['minute']]
	count_df['sunup'] = [check_sunup(timestamp) for timestamp in count_df['audiomoth_timestamp']]

	# Add static columns
	count_df['deployment'] = deployment
	count_df['moth'] = moth
	count_df['box'] = box

	return count_df

def collect_counts(save_directory, source_directory, model, root, moths):
	"""
	Make a counts csv file from each segments csv file generated by collect_segments

	Parameters:

		save_directory: directory containing the vocalization segments by deployment (one csv per deployment with data from all moths)
		model: the name of the model (date and time ID in format yyyymmdd_hhmmss) that generated the segments

	Returns:
		None

	"""

	deployment_dates = load_json(os.path.join(root, 'parameters', 'deployment_dates.json'))
	label_names = load_json(os.path.join(root, 'parameters', 'vocalization_label_names.json'))
	to_process = glob.glob(os.path.join(source_directory, '*segments.csv'))
	
	for path in to_process:  
		print('processing',os.path.split(path)[-1])
		box = [i for i in os.path.split(path)[-1].split('_') if 'box' in i][0]
		deployment = os.path.split(path)[-1].split('_')[0]
		save_path = os.path.join(save_directory, deployment+'_'+box+'_counts.csv')

		if not os.path.exists(save_path):
			segments_df = pd.read_csv(path)
			count_df = get_counts(segments_df, root = root)
			count_df.to_csv(save_path, index=False)
		else:
			print('\tcounts file already generated')
	print('done.')
    
def write_train_job_scripts(models_root, params_path):
    """
    Give model training parameters
    Get an sbatch script for training the model on a GPU
    """

    #make sure paths exist before start
    with open(params_path, 'r') as fp:
        params = json.load_json(fp)

    #write an .sbatch file using the keys in params
    lines = [
    '#!/bin/bash\n', 
    '#\n', 
    '#SBATCH --job-name='+params['job_name']+ '\n',  
    '#SBATCH -p '+params['partition']+ '\n', 
    '#SBATCH -n 1 # one node\n',
    '#SBATCH -t '+params['requested_time']+ '\n',
    '#SBATCH --mem='+params['requested_memory']+ '\n',
    '#SBATCH -o '+params['species']+'_dastrain_%A_%a.out # Standard output\n', 
    '#SBATCH -e '+params['species']+'_dastrain_%A_%a.err # Standard error\n',  
    '#SBATCH --gres=gpu:1\n',
    '\n', 
    '#load the modules and activate your das conda environment\n',
    'module load Anaconda3/5.0.1-fasrc02\n',
    'module load cuda/9.0-fasrc02 cudnn/7.4.1.5_cuda9.0-fasrc01\n',
    '\n',
    'source activate das\n',
    '\n',
    'DATADIR='+ params['training_data_dir'] + '\n',
    'SAVEDIR='+ params['save_dir'] +'\n',
    'NB_EPOCH='+ params['NB_EPOCH'] +'\n',
    'MODEL='+ params['MODEL'] +'\n',
    'KERNEL_SIZE='+ params['KERNEL_SIZE'] +'\n',
    'NB_FILTERS='+ params['NB_FILTERS'] +'\n',
    'NB_HIST='+ params['NB_HIST'] +'\n',
    'NB_PRE_CONV='+ params['NB_PRE_CONV'] +'\n',
    'NB_CONV='+ params['NB_CONV'] +'\n',
    '\n',
    '#train\n',
    'das train --data-dir $DATADIR --save-dir $SAVEDIR --model-name $MODEL --verbose 1 --kernel-size $KERNEL_SIZE --nb-filters $NB_FILTERS --nb-hist $NB_HIST --nb-pre-conv $NB_PRE_CONV --nb-epoch $NB_EPOCH --nb-conv $NB_CONV -i'
    ]

    #write lines
    sbatch_name = 'das_train_'+params['species']+'.sbatch'
    sbatch_save_path = os.path.join(params['job_scripts_dir'], sbatch_name)
    with open(sbatch_save_path, 'w') as f:
        f.writelines(lines)

    print('wrote job scripts to:\n\t', params['job_scripts_dir'])

def get_intersyllable(data, label_column):
	"""
	get the start and stop times of intersyllable intervals

	Arguments:
		data (dataframe): dataframe where each row is a predicted or annotated vocalization with columns for start and stop time in seconds
		label_column (str): name fo the column with the label (cry or USV)
	Reurns:
		squeak_intersyllables (list): list of squeak intersyllable intervals
		USV_intersyllables (list): list ofUSV intersyllable intervals

	"""

	squeak_intersyllables = []
	USV_intersyllables = []


	squeak_df = data.loc[data[label_column] == 'squeak'].reset_index(drop=True)
	USV_df = data.loc[data[label_column] == 'USV'].reset_index(drop=True)

	if len(squeak_df) > 0:
		for voc in range(len(squeak_df) - 1): # time from end of each voc to start of the next (last voc is NaN)
				stop = squeak_df['stop_seconds'].iloc[voc]
				next_start = squeak_df['start_seconds'].iloc[voc + 1]
				squeak_intersyllables.append(next_start - stop)

	if len(USV_df) > 0:
		for voc in range(len(USV_df) - 1):
				stop = USV_df['stop_seconds'].iloc[voc]
				next_start = USV_df['start_seconds'].iloc[voc + 1]
				USV_intersyllables.append(next_start - stop)

	return squeak_intersyllables, USV_intersyllables

def get_wav_clips(wavs_dir, save_location, source_data, margin, start_column, end_column, label_column = None, audiomoth = None, units = 's'):
    """
    Use start and stop times of vocalizations to save individual clips as .wav files (one per detected voc)

    Arguments:
        wavs_dir (string): the path to the raw wav files that have already been segmented
        save_location (string): the path to the directory where the clips should be saved
        source_data (dataframe): should contain at least the columns ['source_file', 'start_times', 'stop_times']
        margin (float): a margin (in seconds) to be added before each start time and after each stop time
        start_column (string): the name of the column in source_data that contains the vocalization start times
        end_column (string): the name of the column in source_data that contains the vocalization stop times
        label_column (string): the name of the column in source_data that contains labels for each vocalization (optional)
        audiomoth (string): optional name of the audiomoth to get wav clips from
        units (string): the temporal units for start and stop times (s or ms)

    Returns
    -------
       None

    """

    #optionally subset by audiomoth
    if audiomoth != None:
        df = source_data.loc[source_data['species'] == species]
    else:
        df = source_data

    #get the names of the recording source files 
    source_files = df['source_file'].unique()

    #for each recording in df, load the wav, subset the big data frame to get just the start and stop times for this recording, then 
    #for each start and stop time (for each clip), get the clip, name it, and write it to save_location. Note that time is assumed
    #to be in ms here.

    already_processed = [i.split('_clip')[0] for i in os.listdir(save_location)]

    for file in source_files:
        
        source_name = file.split('/')[-1] 
        deployment = file.split('/')[-2]
        audiomoth = file.split('/')[-3]
        
        sf_df = df.loc[df['source_file'] == file]
        num_vocs_to_process = len(sf_df)
        num_already_processed = len([i for i in already_processed if source_name.split('.')[0] in i])

        
        if ('_').join([audiomoth, deployment,source_name.split('.')[0]]) in already_processed and num_vocs_to_process==num_already_processed:
            continue

        else:
            path_to_source = file  
            fs, wav = wavfile.read(path_to_source)
            sf_df['clip_number'] = range(num_vocs_to_process)
            count = 0
            print('preparing to get', len(sf_df), 'clips from', file.split('/')[-1])
            for idx, _ in sf_df.iterrows(): 
                start, end = sf_df.loc[idx, (start_column)], sf_df.loc[idx, (end_column)] #get the start and stop time for the clip

                if label_column != None:
                    clip_name = ('_').join([audiomoth, deployment,source_name.split('.')[0],'clip',str(sf_df.loc[idx, 'clip_number']),sf_df.loc[idx, label_column]]) + '.wav'   

                else:
                    clip_name = ('_').join([audiomoth, deployment,source_name.split('.')[0],'clip',str(sf_df.loc[idx, 'clip_number'])])+'.wav' 

                if units == 's':
                    start= int((start - margin)*fs)
                    end =  int((end + margin)*fs)
                    clip = wav[start:end] #get the clip
                    wavfile.write(os.path.join(save_location,clip_name), fs, clip) #write the clip to a wav
                    count+=1

                elif units == 'ms':
                    start, end = start - margin, end + margin
                    start, end = int((start/1000)*fs), int((end/1000)*fs) #convert to sampling units
                    clip = wav[start:end] #get the clip

                    wavfile.write(os.path.join(save_location,clip_name), fs, clip) #write the clip to a wav
                    count+=1

            print(' ...got', num_vocs_to_process,'wav clips')
    print('done.')
    
    



