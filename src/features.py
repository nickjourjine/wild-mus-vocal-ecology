#this file contains functions for calculating acoustic features from wav files

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
import textwrap

from scipy.ndimage.filters import gaussian_filter
from datetime import date, datetime, timedelta

from scipy.signal import stft, welch, butter, lfilter
from scipy.integrate import simps

from src.timestamps import check_sunup, get_box_from_audiomoth, get_deployment_from_time
from src.filespaths import sort_nicely
from src.parameters import load_json



def get_acoustic_diversity_index(wav_path, nperseg=1024, noverlap=1024//4, fmax = 90000):
    """
    use scikit.maad to calculate acoustic diveristy index of a wav file - wav is the path to that file
    from: https://scikit-maad.github.io/generated/maad.features.acoustic_diversity_index.html
    """
    
    from maad import sound, features, rois
    from maad.util import power2dB, plot2d, format_features, overlay_rois, overlay_centroid
    
    #load sound
    s, fs = sound.load(wav_path, detrend=False)
    
    #make spectrogram
    Sxx, tn, fn, ext = sound.spectrogram(s, fs, nperseg= nperseg, noverlap=noverlap, mode='amplitude', detrend=False) 
    
    #get acoustic diversity index
    ADI  = features.acoustic_diversity_index(Sxx,fn,fmax=fmax) #fmax hard coded to max frequency expected for mouse vocalizations
    
    return ADI

def get_bandpower(psd, sf, band, freqs, window_sec=None, relative=False):
##modified from https://raphaelvallat.com/bandpower.html
	"""Compute the average power of the signal x in a specific frequency band.

	Parameters
	----------
	data : 1d-array
		Input signal in the time-domain.
	sf : float
		Sampling frequency of the data.
	band : list
		Lower and upper frequencies of the band of interest.
	window_sec : float
		Length of each window in seconds.
		If None, window_sec = (1 / min(band)) * 2
	relative : boolean
		If True, return the relative power (= divided by the total power of the signal).
		If False (default), return the absolute power.

	Return
	------
	bp : float
		Absolute or relative band power.
	"""

	band = np.asarray(band)
	low, high = band

	# Define window length
	if window_sec is not None:
		nperseg = window_sec * sf
	else:
		nperseg = (2 / low) * sf

	# Frequency resolution
	freq_res = freqs[1] - freqs[0]

	# Find closest indices of band in frequency vector
	idx_band = np.logical_and(freqs >= low, freqs <= high)

	# Find closest indices of band in frequency vector
	idx_band = np.logical_and(freqs >= low, freqs <= high)

	# Integral approximation of the spectrum using Simpson's rule.
	bp = simps(psd[idx_band], dx=freq_res)

	return bp
		
def get_bandpower_batch(list_to_process, lowband, highband, noise_path, num_to_process, nperseg=512, step=2, ignore = None, audio_dir=None):
    """
   1. For each recording in audio_list, step through and calculate energy in psd in one of two bands (highband and lowband)
   2. Get the max of each for each recording
   3. Collect all of these maximums in a dataframe


    Return
    ------
    max_bps : numpy array
        array with shape (2,sample_number) where each item is the areas under curve sample_psd - noise_psd in lowband and highband, respectively
    sample_list: list of str
        the samples that were processed 
    """

    #check inputs
    assert not (list_to_process==None) & (audio_dir==None), "at least one of sample_list or audio_dir must be specified"
    assert not (list_to_process!=None) & (audio_dir!=None), "at most one of sample_list or audio_dir must be specified"
    
    
    #get the psd of the noise sample
    fs, noise_audio = wavfile.read(noise_path)
    noise_freqs, noise_psd = welch(noise_audio, fs, nperseg=nperseg)

    #get the samples to iterate through
    if audio_dir != None:
        sample_list = sort_nicely([audio_dir+i for i in os.listdir(audio_dir) if i.endswith('.wav') and not i.startswith('.')])

        
    #if num_to_process is given, take a random sample 
    if list_to_process == None:
        sample_list = sample_list[ignore[0]:ignore[1]]
    elif list_to_process != None:
        sample_list = list_to_process

    #initialize list for collecting info
    max_bps = []
    processed = []

    #iterate through each sample
    for sample in tqdm(sample_list):
        processed.append(sample)

        #read the audio
        if os.path.getsize(sample) != 0: # don't try to open an empty file in case one gets written
            fs, audio = wavfile.read(sample)
        else:
            continue

        #set time steps and initialize lists for collacting mid and high bandpower
        time_step = step*fs
        times = range(len(audio))[::time_step]
        lows = []
        highs = []

        #step through the audio and get bandpower
        for start in times:
            clip = audio[start:start+time_step]

            #get psd
            sample_freqs, sample_psd = welch(clip, fs, nperseg=nperseg)

            #get the difference between sample and noise
            diff = sample_psd - noise_psd

            #get area under the curve of diff within the bounds provided by lowband and highband
            relative_low = get_bandpower(diff, 
                                         fs, 
                                         band=lowband, 
                                         freqs=sample_freqs,
                                         window_sec=nperseg/fs) 

            relative_high = get_bandpower(diff, 
                                          fs, 
                                          band=highband, 
                                          freqs=sample_freqs,
                                          window_sec=nperseg/fs)

            lows.append(relative_low)
            highs.append(relative_high)

        max_bps.append([np.max(lows), np.max(highs), sample])

    #make a dataframe
    max_bps_df = pd.DataFrame(max_bps, columns = [str(lowband), str(highband), 'source_file'])

    #max_lows
    max_lows = [i[0] for i in max_bps]
    
    #max_highs
    max_highs = [i[1] for i in max_bps]
    
    #return the band power in lowband and highband as a numpy array, return the list of samples processed
    return max_bps_df, max_lows, max_highs
	
#make an array of values rectangular by adding nan (useful for plotting heatmaps)

#threshold an array  
def get_rms(wav_clip, from_spec = False, n_fft=1024):
    """
    Take a wav clip (.wav file of a single vocalization )and calculate the median of the root mean squeared energy 
    for each frame in the clip

    Arguments:
        wav_clip (string): path to the wav clip
        from_spec (bool): if True, get rms from a spectrogram (slower but more accurate). if False, get it from the audio (faster but less accurate)
        frame_length (int): passed to librosa.feature.rms to determine how many frames 
        hop_length (int): passed to librosa.feature.rms to generate spectrogram in from_spec=True
        

    Returns:
        rmse: root mean squared error of the signal in the wav clip
    """

    #TO DO - fix the bug if from_spec=True
    
    #get the wav
    wav, fs = librosa.load(wav_clip)
    
    if from_spec:
        S = np.abs(librosa.stft(wav,  n_fft=n_fft))
        rms = np.median(librosa.feature.rms(S=S, frame_length = n_fft))
        
    else:
        rms = np.median(librosa.feature.rms(y=wav, frame_length=n_fft))

    return rms
def make_selection_tables(root, segments_root, model_ID, annotated = False,ignore_moth00 = True):
	# make and save a warbleR style selection tables for each box for each deployment
	# making multiple like this is necessary because warbleR only operates on batches of wav files stored in a single directory
	# save_dir is where the csvs of the selection tables will be saved

	# storage locations of each deployment
	audio_locations = load_json(os.path.join(root,'parameters', 'recording_storage_locations.json'))

	#get the predictions
	if annotated:
		prediction_paths = glob.glob(os.path.join(segments_root, 'annotations_current', '*.csv'))
	else: 
		prediction_paths = glob.glob(os.path.join(segments_root, '*_segments.csv'))
#		print(segments_root)
		
	for path in tqdm(prediction_paths):

		#get the predictions
		all_predictions = pd.read_csv(path)
		boxes = all_predictions['box'].unique()
		
		#get the boxes
		if annotated: 

			annotation_name = os.path.split(path)[-1]
			date = annotation_name.split('_')[0]
			deployment = get_deployment_from_time(root, date)
			moth = 'audio'+annotation_name.split('_')[2]
			box = get_box_from_audiomoth(root, moth, deployment)
			save_path = os.path.join(segments_root, 'annotations_current_features', 'selection_tables', annotation_name)
			wavs_dir = os.path.join(segments_root, 'annotations_current_features', 'wavs')

			if not os.path.exists(save_path):
				selection_table = pd.read_csv(path)
				
				# remove labels that don't have any annotations
				if selection_table['stop_seconds'][selection_table['name'] == 'cry'].sum() == 0:
					selection_table = selection_table[selection_table['name'] != 'cry']
				if selection_table['stop_seconds'][selection_table['name'] == 'USV'].sum() == 0:
					selection_table = selection_table[selection_table['name'] != 'USV']
				
				# make the selection table
				selection_table = selection_table.rename(columns = {'name':'label', 'start_seconds':'start', 'stop_seconds':'end'})
				selection_table['sound.files'] = annotation_name.replace('_annotations.csv', '.wav')
				selection_table['deployment'] = deployment
				selection_table['moth'] = moth
				selection_table['box'] = box
				selection_table['full.path'] = os.path.join(wavs_dir, annotation_name.replace('_annotations.csv', '.wav'))
				selection_table['wavs.dir'] = wavs_dir
				selection_table['selec'] = selection_table.groupby('sound.files').cumcount() + 1

				assert sorted(selection_table.columns) == sorted(['sound.files', 'selec','start', 'end', 'label', 'deployment', 'moth', 'box', 'full.path', 'wavs.dir']), "Your selection table has incorrect columns..."

				#save
				if 'selection_tables' not in os.listdir(segments_root):
					os.mkdir(os.path.join(segments_root, 'selection_tables'))

				if ignore_moth00:
					if not moth=='audiomoth00':
						selection_table.to_csv(save_path, index=False)
				else:
					selection_table.to_csv(save_path, index=False)

			else:
				print(save_path, 'already exists...')
			
		
		else:
			
			for box in boxes:

				#get the predictions for this box and make sure you have one box on one deployment
				predictions = all_predictions[all_predictions['box'] == box]
				assert len(predictions['box'].unique()) == 1
				assert len(predictions['deployment'].unique()) == 1
				assert len(predictions['moth'].unique()) == 1
				deployment = predictions['deployment'].unique()[0]
				moth = predictions['moth'].unique()[0]

				#check if you made the file already
				save_path = os.path.join(segments_root,'selection_tables',('_').join([deployment, moth, 'box'+str(box),'selectiontable'])+'.csv')
				
				if not os.path.exists(save_path):

					# ignore the predictions labeled as nosie
					predictions = predictions[predictions['label'] != 'noise']

					#rename old column names
					if 'minute' in predictions.columns:
						predictions = predictions.rename(columns = {'minute': 'audiomoth_timestamp'})

					# drop any duplicates (there shouldn't be any)
					predictions = predictions.drop_duplicates(subset=['audiomoth_timestamp', 'start_seconds', 'stop_seconds'], ignore_index = True)

					#add columns for warbleR
					predictions['full.path'] = [os.path.join(audio_locations[d], m, ('_').join([d,'box'+str(b)]), f+'.wav') for d,m,b,f in       zip(predictions['deployment'], predictions['moth'], predictions['box'], predictions['audiomoth_timestamp'])]
					predictions['wavs.dir'] = [os.path.split(i)[0] for i in predictions['full.path']]
					predictions['sound.files'] = [os.path.split(i)[-1] for i in predictions['full.path']]
					predictions['selec'] = predictions.groupby('sound.files').cumcount() + 1

					#make the selection table
					predictions = predictions.rename(columns={'start_seconds':'start', 'stop_seconds':'end'})
					selection_table = predictions[['sound.files', 'selec','start', 'end', 'label', 'deployment', 'moth', 'box', 'full.path', 'wavs.dir']]

					#save
					if 'selection_tables' not in os.listdir(segments_root):
						os.mkdir(os.path.join(segments_root, 'selection_tables'))

					if ignore_moth00:
						if not moth=='audiomoth00':
							selection_table.to_csv(save_path, index=False)
							print(save_path)
					else:
						selection_table.to_csv(save_path, index=False)

				else:
					print(save_path, 'already exists...')
				
def write_feature_extraction_script(segments_root, selection_tables_dir, features_save_path, script_save_path, project_root, wl, ovlp, bp, mar, ncores, write_file = False, annotation = False):
	"""
	Write and save an R script that iterates through selection tables and calculates features using the features.R file
	"""

	# Get the selection tables    
	#selection_tables_dir = os.path.join(segments_root, segment_type, 'selection_tables')
	#assert os.path.exists(selection_tables_dir), "It looks like you haven't made your selection tables yet. Do this First."

	# make a folder to save features if it doesn't exist
	if not os.path.exists(features_save_path):
	    os.makedirs(features_save_path)
	    print('...made a directory to save features at', features_save_path)
		
	# change to R boolean 
	if annotation:
		annotation = 'T'
	else:
		annotation = 'F'
		
	

	# Create the R script content
	r_script = """
    # source the R functions you need
    source(file.path('{0}', 'src', 'features.R'))

    # get the selection tables
    selection.tables <- list.files('{1}')

    # iterate through selection tables and calculate features
    for (selection.table in selection.tables) {{
        path.to.selectiontable <- file.path('{1}', selection.table)
        
        # Check if the feature file already exists
        feature_filename <- sub('selectiontable', 'features', selection.table)
        feature_file <- file.path('{2}', feature_filename)
        if (!file.exists(feature_file)) {{
			cat(paste0('Processing ', path.to.selectiontable, '.\\n'))
            get_acoustic_features(path.to.selectiontable, 
                                  save.dir = '{2}',
                                  wl = {3}, 
                                  ovlp = {4}, 
                                  bp = c({5}, {6}), 
                                  mar = {7}, 
								  ncores = {8}, 
								  annotation = {9})
        }} else {{
            cat(paste0('Features already extracted for ', selection.table, '. Skipping.\\n'))
        }}
    }}
	""".format(project_root, selection_tables_dir, features_save_path, wl, ovlp, bp[0], bp[1], mar, ncores, annotation)

	# format
	r_script = textwrap.dedent(r_script)

	if write_file:
		# Save the R script to the specified path
		with open(script_save_path, 'w') as script_file:
			script_file.write(r_script)
		print('...wrote an R script to', script_save_path)

	return (r_script)

def write_train_ava_script(script_save_dir, ava_root, split, model_save_dir, epochs, test_freq, save_freq, vis_freq, model_checkpoint = None, partition = None):
	script_content = f"""
# Import packages
from ava.models.vae_dataset import get_syllable_partition
from ava.models.vae_dataset import get_syllable_data_loaders
from ava.models.vae import VAE
import os

# Paths
ava_root = '{ava_root}'

# Recordings
recordings = os.listdir(ava_root)

# Get the paths to relevant directories
spec_dirs = [os.path.join(ava_root, recording, 'specs') for recording in recordings]

# 80/20 train/test split
split = {split}

# Construct a random train/test partition
partition = get_syllable_partition(spec_dirs, split)

# Make Dataloaders
loaders = get_syllable_data_loaders(partition)

# Construct network
model_save_dir = '{model_save_dir}'
model_checkpoint = '{model_checkpoint}'

# initialize model
model = VAE(save_dir=model_save_dir)

if model_checkpoint == None:
        print("Training new model from scratch and saving to:", model_save_dir) 
else:
        print("Training model from checkpoint:",  model_checkpoint)
        model.load_state(model_checkpoint)

# Train
model.train_loop(loaders, epochs={epochs}, test_freq={test_freq}, save_freq={save_freq}, vis_freq={vis_freq})
"""

	# Save the script to the specified directory
	script_file = os.path.join(script_save_dir, 'train_vae.py')
	with open(script_file, 'w') as f:
		f.write(script_content)
	print(f"Generated script saved to {script_file}")
	return script_content

    
    

    
    
    

    
    