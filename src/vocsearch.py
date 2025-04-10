# this file contains functions for finding recordings with vocalizations to use for training data

from scipy.io import wavfile
from scipy.signal import stft
import matplotlib.pyplot as plt
from scipy.signal import stft, welch, butter, lfilter
from scipy.integrate import simps
import numpy as np
import re
import os
import glob
import random
from tqdm import tqdm
import pandas as pd

#this is your utilities.py file - see https://goodresearch.dev/setup.html for how you did this
from src import filespaths, features, annotation

def get_recordings_from_psd(df, save_dir, thresh):
    """
    take a df of power spectral density and return text files listing 
    the recording chunks that have the loudest 15-35 and 55-75 Khz bandpower
    """
    
    #get recordings passing 55-75 Khz bandpower threshold (and average recordings or comparison just in case)
    high_loud =  df['source_file'].loc[df['[55000, 75000]'] > thresh*np.std(df['[55000, 75000]']) + np.mean(df['[55000, 75000]'])]
    high_middle = df['source_file'].loc[(np.mean(df['[55000, 75000]']) - 0.05*np.std(df['[55000, 75000]']) < df['[55000, 75000]']) & (df['[55000, 75000]'] < 0.05*np.std(df['[55000, 75000]']) + np.mean(df['[55000, 75000]']))]
    
    #make lists for writing
    high_to_write = ["loud -"]+[str(thresh)]+[" std above mean"]+list(high_loud)+['middle - within 0.05 std of mean']+list(high_middle)
    high_to_write = ("\n").join([i for i in high_to_write])
    
    #write
    high_save_name = ('_').join(['55-75Khz_recordings_of_interest',os.path.split(df['source_file'][0])[0].split('/')[-1]])+'.txt'
    with open(os.path.join(save_dir, high_save_name), 'w') as fp:
        for line in high_to_write:
            fp.write(line)
        
    #get recordings passing 15-35 Khz bandpower threshold (and average recordings or comparison just in case)
    low_loud =  df['source_file'].loc[df['[15000, 35000]'] > 2*np.std(df['[15000, 35000]']) + np.mean(df['[15000, 35000]'])]
    low_middle = df['source_file'].loc[(np.mean(df['[15000, 35000]']) - 0.05*np.std(df['[15000, 35000]']) < df['[15000, 35000]']) & (df['[15000, 35000]'] < 0.05*np.std(df['[15000, 35000]']) + np.mean(df['[15000, 35000]']))]
    
    #make lists for writing
    low_to_write = ["loud -"]+[str(thresh)]+[" std above mean"]+list(low_loud)+['middle - within 0.05 std of mean']+list(low_middle)
    low_to_write = ("\n").join([i for i in low_to_write])
    
    #write
    low_save_name = ('_').join(['15-35Khz_recordings_of_interest',os.path.split(df['source_file'][0])[0].split('/')[-1]])+'.txt'
    with open(os.path.join(save_dir, low_save_name), 'w') as fp:
        for line in low_to_write:
            fp.write(line)
            
    print('saved recordings of interest to', save_dir)
    

#related to file naming

def get_recordings_from_das(df, save_dir, cry_thresh, USV_thresh, audio_root, annotation_iteration):
    """
    take a df of DNN predictions and return a text file listing the recordings with more than cry_thresh cries and USV_thresh USVs
    """
    
    #make sure there is only one deployment and audiomoth in the df
    assert len(set(df['deployment'].unique())) == 1, "There is more than one deployment in the dataframe (there should be only one)"
    assert len(set(df['box'].unique())) == 1, "There is more than one box in the dataframe (there should be only one)"
    assert len(set(df['moth'].unique())) == 1, "There is more than one audiomoth in the dataframe (there should be only one)"
    
    deployment = list(df['deployment'].unique())[0]
    box = 'box'+str(list(df['box'].unique())[0])
    
    #add a source_file and cry/USV count column
    df['source_file'] = [os.path.join(audio_root, moth, deployment+'_box'+str(box), minute+'.wav') for moth, deployment, box, minute in zip(df['moth'], df['deployment'], df['box'], df['minute'])]
    df['cry'] = [1 if i==2 else 0 for i in df['label']]
    df['USV'] = [1 if i==1 else 0 for i in df['label']]
    
    #get counts for each file
    to_annotate = df[['source_file', 'cry', 'USV']].groupby(['source_file']).sum()
    to_annotate['source_file'] = to_annotate.index
    to_annotate = to_annotate.reset_index(drop=True)
    
    #drop any files that have already been annotated
    to_annotate['annotated_already'] = [annotation.check_annotated(i, annotation_iteration) for i in to_annotate['source_file']]
    to_annotate = to_annotate.loc[to_annotate['annotated_already'] == 0]
    
    to_annotate['model'] = df['model'][0]
    to_annotate = to_annotate.loc[(to_annotate['cry'] >= 1)&(to_annotate['USV'] >= 1)]
    to_annotate = to_annotate.sort_values(by='USV', ascending=False)

    #write a CSV to that you can use to choose recordings for annotation
    to_annotate.to_csv(os.path.join(save_dir, ('_').join([deployment, box])+'.csv'), index=False)
    
    print('saved', len(to_annotate) ,'recordings of interest to', save_dir)
    

#related to file naming

def get_clip(wav_path,start,stop, units):
    "clip a wav file into smaller wav file"
    #check the inputs
    assert os.path.exists(wav_path)
    assert stop > start
    assert units in ['s', 'ms']
    
    #get the wav
    fs, wav = wavfile.read(wav_path)
    
    #clip it
    if units == 's':
        start, stop = start*fs, stop*fs
        clip = wav[start:stop]
        return clip
    elif units == 'ms':
        start, stop = (start/1000)*fs, (stop/1000)*fs
        clip = wav[start:stop]
    
    return clip
    
    
def get_background_example(moth, raw_dir, save_dir, margin=0,duration=None, units = 's', fs=192000):
    """
    Interactive function to get a region of recording for each deployment that does not contain vocalizations.

    Parameters
    ----------
    moth (str): audiomoth ID (one of [audiomoth00, audiomoth01, audiomoth02, audiomoth03, audiomoth04])

    raw_dir (string): the path to the directory containing the raw recordings for which you want to get noise

    save_dir (string): the path to the directory where the noise clips should be saved (as .wav)

    margin (float): a margin in seconds to add before the start and after the end of each noise clip

    min_dur: the minimum duration in seconds of the desired noise clips

    max_dur: the maximum duration in seconds of the desired noise clips

    units ('s' or 'ms'): units in which time is measured

    Returns
    -------
    None

    """

    deployment = os.path.split(raw_dir)[-1]
    save_name = ('_').join([moth,deployment])+'.wav'

    assert os.path.exists(save_dir)
    assert fs == 192000
    assert save_name not in os.listdir(save_dir)
    assert duration != None
    assert 1 < duration < 10

    #pick a file at random from this moth deployment
    wav_path = random.sample(glob.glob(raw_dir+'*.wav'), k=1)[0]

    #pick 10 start and stop time pairs with givne duration
    starts = [int(i) for i in np.linspace(0,45,10)]
    stops = [i+duration for i in starts]
 
    #show each clip
    for start,stop in zip(starts,stops):
        clip = get_clip(wav_path=wav_path,start=start,stop=stop,units=units)
        t,f,spec = stft(clip, noverlap=256, nperseg=1024, fs=fs)
        spec = np.log(np.abs(spec))
        plt.figure(figsize=[5,5])
        plt.imshow(spec, origin='lower')
        plt.show()

        #get input 

        val = input("ok to use as a background nonvocal example? (y/n/exit)")
        assert val in ['y', 'n', 'exit']
        if val == 'y':
            save_name = ('_').join(wav_path.split('/')[-3:]).split('.')[0]+'_background_example.wav'
            prefix = ('_').join(wav_path.split('/')[-3:-1])
            if not np.any([i.startswith(prefix) for i in os.listdir(save_dir)]):
                print('saving clip...')
                wavfile.write(save_dir + save_name, fs, clip) #write the clip to a wav
                return
            else:
                print('clip already exists for this deployment...')
                return
        elif val == 'n':
            continue
        elif val =='exit':
            return

    print('done.')


def threshold(array,threshold):
    """
    set to 0 each item in an array less than a threshold
    """
    array[array<threshold] = 0
    return array
    
#get intersyllable intervals for merging 	
def get_intersyllable_intervals(df):
	import seaborn as sns
	#predictions_df = annotations
	all_intersyllables = []
	files = []
	source_files = [i for i in df['source_file'].unique()]
	for file in source_files:   
	
		#get the predictions
		test_df = df.loc[df['source_file'] == file]
	
		#get the starts and stops
		starts = list(test_df['start_seconds'])
		stops = list(test_df['stop_seconds'])
		start_or_stop = ['start']*len(starts)+['stop']*len(stops)

		#sort them so they are alternating
		intersyllable = pd.DataFrame()
		intersyllable['times'] = starts + stops
		intersyllable['start_or_stop'] = start_or_stop
		intersyllable = intersyllable.sort_values(by='times')
		
	
		#take the difference between consecutive rows - half of these are the intersyllable intervals (every other)
		intersyllables =  list(intersyllable['times'].diff())[::2]
	
		all_intersyllables.extend(intersyllables)
		files.extend([file]*len(intersyllables))

	all_intersyllables_df = pd.DataFrame()
	all_intersyllables_df['source_file'] = files
	all_intersyllables_df['intersyllable'] = all_intersyllables
	
	return all_intersyllables_df
