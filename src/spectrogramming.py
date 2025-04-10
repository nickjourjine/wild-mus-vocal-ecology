#this file contains functions for making and visualzing spectrograms

#filesystem
import os
import glob
from tqdm import tqdm
from datetime import datetime

#plotting
import seaborn as sns 
import matplotlib.pyplot as plt

#data
#import umap ## causing circular import error
import time
import random
import numpy as np
import pandas as pd

from scipy.io import wavfile
from scipy.signal import stft
from scipy.interpolate import interp2d
from sklearn.preprocessing import StandardScaler

#custom modules
from src.parameters import save_json
from src.timestamps import make_filename_timestamp, audiomoth_to_datetime, get_season_from_date

def make_spec(audio, fs, nperseg, noverlap, thresh, scaling_factor = None):
    """
    Make a spectrogram 
    scaling_factor is a factor for log_resizing durations (following code shared by Tim Sainburg)
    thresh is the minimum pixel value

    """

    #get the spectrogram
    f,t,specgram = stft(audio, fs, nperseg=nperseg, noverlap=noverlap) #default winow is Hann

    #make it pretty
    specgram = np.abs(specgram) #remove complex
    specgram = np.log(specgram)  # take log

    if scaling_factor!= None:
        specgram = log_resize_spec(specgram, scaling_factor= scaling_factor)    
        specgram[specgram < thresh] = thresh
        specgram = (specgram - np.min(specgram)) / (np.max(specgram) - np.min(specgram))

    else: 
        specgram[specgram < thresh] = thresh
        specgram = (specgram - np.min(specgram)) / (np.max(specgram) - np.min(specgram))

    return f,t,specgram

def make_square_spec(data, fs, nperseg, noverlap, num_freq_bins, num_time_bins, min_freq, max_freq, fill_value, max_dur, spec_min_val, spec_max_val): 
    """
    Make a spectrogram with a pre-determined number of frequency and time bins. Useful for generating spectrograms that are all the same shape.
    Modified from: https://autoencoded-vocal-analysis.readthedocs.io/en/latest/_modules/ava/preprocessing/utils.html?highlight=spec
    Used to be called get_spectrogram()

    Arguments:
        data (numpy array): the wav file you want to make a spectrogram from
        nperseg (int): nperseg for spectrogram generation
        noverlap (int): noverlap for spectrogram generation with scipy.
        num_freq_bins (int): number of frequency bins for spectrogram
        num_time_bins (int): number of time bins for spectrogram
        min_freq (int): minimum frequency for spectrogram
        max_freq (int): maximum frequency for spectrogram
        fill_value (float): the value to use for points outside of the interpolation domain (for scipy.interpolate.interp2d)
        max_dur (float): the duration of the longest vocalization you want to consider - determines the time axis scaling
        spec_min_val (float): maximum spectrogram value
        spec_max_vale (float): minimum spectogram value

    Returns
    -------
        f (numpy array): the frequency bins of the spectrogram
        t (numpy array): the time bins of the spectrogram
        specgram: the spectrogram
    """

    #get the spectrogram
    f,t,specgram = stft(data, fs, nperseg=nperseg, noverlap=noverlap) #default winow is Hann

    #define the target frequencies and times for interpolation
    duration = np.max(t)
    target_freqs = np.linspace(min_freq, max_freq, num_freq_bins)
    shoulder = 0.5 * (max_dur - duration)
    target_times = np.linspace(0 - shoulder, duration+shoulder, num_time_bins)

    #process
    specgram = np.log(np.abs(specgram)+ 1e-12) # make a log spectrogram 
    interp = interp2d(t, f, specgram, copy=False, bounds_error=False, fill_value=fill_value) #define the interpolation object 
    target_times = np.linspace(0 - shoulder, duration+shoulder, num_time_bins) #define the time axis of the spectrogram
    interp_spec = interp(target_times, target_freqs, assume_sorted=True) #interpolate 
    specgram = interp_spec
    specgram -= spec_min_val #normalize
    specgram /= (spec_max_val - spec_min_val) #scale
    specgram = np.clip(specgram, 0.0, 1.0) #clip

    return f,t,specgram

def specs_to_umap(specs_list, clip_names, features_df, save_dir, save, spec_params, ):
	"""
	Give a list of numpy arrays (spectrograms), clip_names_list, a list of their names in the same order as the specs, save_root, a path to save the embedding output if save is true. Get a csv with one row per vocalization and columns for embedding coordinates
	"""

	#linearize
	specs_lin, shape = linearize_specs(specs_list)
	del specs_list #free up space

	#zscore
	df_umap, zscored = zscore_specs(specs_lin, clip_names)
	del specs_lin #free up space

	#embed
	umap1, umap2 = get_umap(zscored)
	del zscored #free up space

	#make a directory to save if you haven't already
	print('saving umap coordinates...')

	#name the params and coordinates
	now = make_filename_timestamp()
	coordinates_save_name = ('_').join([now, 'UMAPembedding.feather'])
	params_save_name = ('_').join([now, 'spec_params_for_UMAPembedding.json'])

	#add the coordinates to df_umap
	file_format = '.feather'
	df_umap = df_umap.reset_index(drop=True)
	df_umap.columns = df_umap.columns.map(str)
	df_umap['umap1'] = umap1
	df_umap['umap2'] = umap2

	# merge with selection table and add some useful info
	merged_df = df_umap.merge(features_df, on='clip_name', how = 'left').reset_index(drop=True)
	merged_df['date'] = [audiomoth_to_datetime(i.split('.')[0]) for i in merged_df['sound.files']]
	merged_df['time'] = [i.time() for i in merged_df['date']]
	merged_df['season_code'] = [get_season_from_date(date) for date in merged_df['date']]
	merged_df['season'] = merged_df['season_code'].map({0:'winter', 1:'spring', 2:'summer', 3:'autumn'})

	#save
	if save:
		merged_df.to_feather(os.path.join(save_dir,coordinates_save_name))
		save_json(spec_params, save_dir = save_dir, save_name = params_save_name)
		return merged_df
	else:
		return merged_df
	
	
	
def selection_table_to_umap(root, table, spec_params, num_to_process, save_root, use_dir = None ,verbose=True, save = True):
	"""
	Take a warbleR style selection table, generate spectrograms from each clip in the table, then find a umap embedding fo those spectrograms and save the coordinates 

	Arguments:
		table (dataframe): A warbleR style selection table (e.g. generated by warbleR::spectral_analysis)
		spec_params (dict): dictionary of parameters for generating spectrograms
		num_to_process (int or 'all'): Number to process. If 'all' process everything. Useful for debugging to set this to 10 or 20 first.
		save_root (str): path to the directory where the umap coordinates and spectrograms will be saved

	Returns:
		None

	"""


	if use_dir == None:
		#make directories
		now = make_filename_timestamp()
		coordinates_save_dir = os.path.join(save_root,'umap_coordinates')
		print(coordinates_save_dir)
		print(os.path.exists(coordinates_save_dir))
		if not os.path.exists(coordinates_save_dir):
			os.mkdir(coordinates_save_dir)
		os.mkdir(os.path.join(coordinates_save_dir, now))
		coordinates_save_dir = os.path.join(coordinates_save_dir, now)

	else:
		coordinates_save_dir = use_dir
		print('Starting from where you left off in:\n\t', coordinates_save_dir)
	if use_dir:
		done_files = glob.glob(os.path.join(use_dir, 'spectrograms', '*.npy'))
		done_clip_names = sorted([os.path.split(i)[-1].split('.')[0] for i in done_files])
		done_files_sorted = sorted(done_files, key=lambda x: os.path.split(x)[-1].split('.')[0])
		print('You have made', len(done_files_sorted), 'of', len(table), 'spectrograms...\n')
		table = table[~table['clip_name'].isin(done_files_sorted)]

	#get the spectrograms
	new_specs_list, new_clip_names = specs_from_selection_table(root = root, 
																table = table, 
																spec_params = spec_params, 
																num_to_process = num_to_process, 
																verbose = verbose, 
																save = save, 
																save_dir = coordinates_save_dir)

	#add to existing, ensuring clip names and spectrograms are in the same order
	print('getting done spectrograms...')
	done_specs_list = [np.load(file) for file in done_files_sorted]
	new_clip_names_sorted, new_specs_list_sorted = zip(*sorted(zip(new_clip_names, new_specs_list)))

	print('combining them with new spectrograms...')
	specs_list = done_specs_list + list(new_specs_list_sorted)
	clip_names = done_clip_names + list(new_clip_names_sorted)

	#linearize
	specs_lin, shape = linearize_specs(specs_list)
	del specs_list #free up space

	#zscore
	df_umap, zscored = zscore_specs(specs_lin, clip_names)
	del specs_lin #free up space

	#embed
	umap1, umap2 = get_umap(zscored)
	del zscored #free up space

	#make a directory to save if you haven't already
	print('saving umap coordinates...')

	#name the params and coordinates
	now = make_filename_timestamp()
	coordinates_save_name = ('_').join([now, 'UMAPembedding.feather'])
	params_save_name = ('_').join([now, 'spec_params_for_UMAPembedding.json'])

	#add the coordinates to df_umap
	file_format = '.feather'
	df_umap = df_umap.reset_index(drop=True)
	df_umap.columns = df_umap.columns.map(str)
	df_umap['umap1'] = umap1
	df_umap['umap2'] = umap2

	# merge with selection table and add some useful info
	merged_df = df_umap.merge(table, on='clip_name').reset_index(drop=True)
	merged_df['date'] = [audiomoth_to_datetime(i.split('.')[0]) for i in merged_df['sound.files']]
	merged_df['time'] = [i.time() for i in merged_df['date']]
	merged_df['season_code'] = [get_season_from_date(date) for date in merged_df['date']]
	merged_df['season'] = merged_df['season_code'].map({0:'winter', 1:'spring', 2:'summer', 3:'autumn'})

	#save
	if save:
		merged_df.to_feather(os.path.join(coordinates_save_dir,coordinates_save_name))
		save_json(spec_params, save_dir = coordinates_save_dir, save_name = params_save_name)
		return merged_df, spec
	else:
		return merged_df, spec

    
def specs_from_selection_table(root,  table, spec_params, num_to_process='all', save_dir = None, verbose=True, save = True):
	"""
	Generate spectrograms from a warbleR style selection table (one spectrogram per row of the rable)

	Arguments:
		table (df): a pandas data frame in the format of a warbleR selection table
		spec_params (dict): dictionary of parameters for generating spectrograms
		num_to_process (int or 'all'): Number to process. If 'all' process everything. Useful for debugging to set this to 10 or 20 first.

	Returns:
		specs_list (list): a list of spectrograms as numpy arrays
		source_files (list): a list of file names for each spectrogram
	"""

	# check if a directory for spectrograms already exists
	if save:
		if not os.path.exists(save_dir):
			os.mkdir(save_dir)
	
	#make sure the table is sorted
	table = table.sort_values(by=['full.path', 'start'])

	#get the paths to the wav files (full recordings, not vocalization clips)
	paths_to_wavs = table['full.path'].unique()

	#initialize the list to collect the spectrograms
	specs_list = []
	clip_names = []
	if verbose:
			print('getting spectrograms...')
			
	iterable = tqdm(paths_to_wavs) if verbose else paths_to_wavs

	for path_to_wav in iterable:

		#load the file
		fs, wav_full = wavfile.read(path_to_wav)

		#get the start and stop of each vocalization 
		segments = table[table['full.path'] == path_to_wav][['start','end','full.path','selec', 'clip_name']]
		
		#remove clips that you already have spectrograms for
		if save:
			done_specs = [i.split('.')[0] for i in os.listdir(save_dir)]
			#print(len(segments[segments['clip_name'].isin(done_specs)])/len(segments), "% done specs from this file...")
			segments = segments[~segments['clip_name'].isin(done_specs)]

		if len(segments)==0:
			continue
		elif (save) and (len(segments)!=0) and ((len(segments[segments['clip_name'].isin(done_specs)])/len(segments)) == 1):
			continue
		else:
			#check the segments are consecutive
			if len(segments) > 1:
				assert (segments['start'].diff() > 0).any(), "vocalizations are not consecutive"

			#for each segment, get the sound clip, make the spectrogram, and append it to the list
			for start, end in zip(segments['start'], segments['end']):

				#check end follow start
				assert (end-start) > 0, "vocalization end precedes start"

				#get the wav clip
				wav_clip = wav_full[int(start*fs):int(end*fs)]

				#name the wav clip
				moth = path_to_wav.split('/')[-3].split('.')[0]
				file = path_to_wav.split('/')[-1].split('.')[0]
				clip_number = list(segments['selec'].loc[(segments['start'] == start) & (segments['end'] == end)])[0]
				clip_name_old = ('_').join([moth,file,'clip'+str(clip_number)])
				clip_name = segments['clip_name'].iloc[0]
				print(clip_name)

				#clip_number = clip_name.split('_')[-1]
				
				#make the spectrogram
				f, t, spec = make_square_spec(data = wav_clip,
											  fs=spec_params['fs'],
											  nperseg=spec_params['nperseg'],
											  noverlap=spec_params['noverlap'],
											  num_freq_bins=spec_params['num_freq_bins'],
											  num_time_bins=spec_params['num_time_bins'],
											  min_freq=spec_params['min_freq'],
											  max_freq=spec_params['max_freq'],
											  fill_value = spec_params['fill_value'],
											  max_dur=spec_params['max_duration'],
											  spec_min_val=spec_params['spec_min_val'], 
											  spec_max_val=spec_params['spec_max_val'])
				
				specs_list.append(spec) #add the spec
				clip_names.append(clip_name) #add the name
				clip_number+=1
					
				if save:
					filename = os.path.join(save_dir, clip_name+".npy")
					np.save(filename, spec)

					

	if verbose:
		print('Done.\n')

	return specs_list, clip_names

def wavs_to_umap(clips_to_process, spec_params, num_to_process, save_root):

	"""
	Take a list of wav files, generate spectrograms from those files, then find a umap embedding for those spectrograms and save the coordinates

	Arguments:
		clips_to_process (list): optional list of paths to wav files if you want to just process files in a given list
		spec_params (dict): dictionary of parameters for generating spectrograms
		num_to_process (int or 'all'): Number to process. If 'all' process everything. Useful for debugging to set this to 10 or 20 first.
		save_root (str): path to the directory where the umap coordinates will be saved

	Returns:
		None

	"""

	#get spectrograms
	specs_list, source_files = specs_from_wavs(clips_to_process = clips_to_process, spec_params=spec_params, num_to_process = num_to_process)

	#linearize
	specs_lin, shape = linearize_specs(specs_list)
	del specs_list #free up space
	print(shape)

	#zscore
	df_umap, zscored = zscore_specs(specs_lin, clip_names)
	del specs_lin #free up space

	#embed
	umap1, umap2 = get_umap(zscored)
	del zscored #free up space

	#plot 
	print('plotting...')
	fig = plt.figure(figsize=(10,10))
	ax = fig.add_subplot(1,1,1)
	ax = plt.scatter(
		umap1,
		umap2,
		c = 'k',
		s = 1,
		alpha = .25, 
		cmap='viridis')
	plt.show()

	#save
	print('saving umap coordinates...')
	df_umap = df_umap.reset_index(drop=True)
	coordinates_save_dir = os.path.join(save_root,'umap_coordinates')
	if 'umap_coordinates' not in os.listdir(save_root):
		os.mkdir(coordinates_save_dir)

	coordinates_save_name = ('_').join([save_root.split('/')[-2],save_root.split('/')[-1],'embedding.feather'])
	save_name = os.path.join(coordinates_save_dir,coordinates_save_name)
	df_umap.columns = df_umap.columns.map(str)
	df_umap['umap1'] = umap1
	df_umap['umap2'] = umap2
	df_umap.to_feather(save_name)
	print('done.')

def specs_from_wavs(clips_to_process, spec_params, num_to_process):
    """
    Generate spectrograms from a list of wav files - useful for umap embedding

    Arguments:
        clips_to_process (list): list of paths to wav files to process
        spec_params (dict): dictionary of parameters for generating spectrograms
        num_to_process (int or 'all'): Number to process. If 'all' process everything. Useful for debugging to set this to 10 or 20 first.

    Returns:
        specs_list (list): a list of spectrograms as numpy arrays
        source_files (list): a list of file names for each spectrogram
    """
    specs_list = []
    source_files = []

    if num_to_process != 'all':
        to_process = clips_to_process[0:num_to_process]
        print('processesing', num_to_process, 'clips')
    else:
        to_process = clips_to_process

    print('making spectrograms...')
    for path in tqdm(to_process):
        source_files.append(path.split('/')[-1])

        #get the wav
        fs, wav = wavfile.read(path)

        f, t, spec = get_spectrogram(data = wav,
                                     fs=spec_params['fs'],
                                     nperseg=spec_params['nperseg'],
                                     noverlap=spec_params['noverlap'],
                                     num_freq_bins=spec_params['num_freq_bins'],
                                     num_time_bins=spec_params['num_time_bins'],
                                     min_freq=spec_params['min_freq'],
                                     max_freq=spec_params['max_freq'],
                                     fill_value = spec_params['fill_value'],
                                     max_dur=spec_params['max_duration'],
                                     spec_min_val=spec_params['spec_min_val'], 
                                     spec_max_val=spec_params['spec_max_val'])

        specs_list.append(spec) #downsample time and frequency

    print('done.')
    return specs_list, source_files

def linearize_specs(specs_list):
    """
    Linearize each spectrogram in a list. For UMAP prepprocessing.

    Arguments:

        specs_list (list): a list of spectrograms as numpy arrays (eg, output of specs from wavs)

    Returns:
        shape (numpy array): the shape of np.array(specs list)
        specs_lin (list): a list of linearized spectrograms as numpy arrays
    """

    #print some useful info
    specs = np.array(specs_list)
    shape = np.shape(specs)
#    print('shape of spectrograms array is:', np.shape(specs))

    #linearize
    # print('linearizing spectrograms...')
    num_features = specs.shape[-1]*specs.shape[-2]
    specs_lin = specs.reshape([-1, num_features])
#    print('done.')

    return specs_lin, shape

def zscore_specs(specs_lin, source_files):
    """
    Zscore each spectrogram in a list. For UMAP preprocessing.

    Arguments:
        specs_lin (list): a list of linearized spectrograms as numpy arrays (eg, output of linearize_specs)

    Returns:
        df_umap (data frame): a dataframe where each row is a spectrogram and each column is a pixel (plus a source_file column with the path to the wav)
        zscored (numpy array): an array of zscored, linearized spectrograms for umap embedding
    """


    #make a dataframe
    df_umap = pd.DataFrame(specs_lin)
    df_umap['clip_name'] = source_files

    # Z-score the spectrogams
#    print('z scoring...')
    zscored = StandardScaler().fit_transform(df_umap.drop(columns=['clip_name']))	
#    print('done.')
    return df_umap, zscored

def get_umap(standardized_features):
	"""
	Find a umap embedding for a set of spectrograms

	Arguments:
		standardized_features (numpy array): an array of linearized, zscored spectrograms (eg output of zscore specs)

	Returns:
		umap1 (numpy array): x coordinates of umap embedding for each vocalization
		umap2 (numpy array): y coordinares of umap embedding for each vocalization
	"""

	import umap 

	#find an embedding
	print('finding a umap embedding...')

	start_time = time.time()
	reducer = umap.UMAP()
	embedding = reducer.fit_transform(standardized_features)
	print('embedding took', time.time() - start_time, 'seconds.')

	umap1 = embedding[:, 0]
	umap2 = embedding[:, 1]

	return umap1, umap2

def save_umap(df_umap, umap1, umap2, save_name = None, file_format = 'feather'):
    """
    Save coordinates of a umap embedding and their corresponding source files

    Arguments:
        df_umap (dataframe): dataframe of non-zscored spectrograms (eg output of zscore specs)
        umap1 (numpy array): x coordinates of umap embedding for each vocalization
        umap2 (numpy array): y coordinares of umap embedding for each vocalization
        save_name (str): the path to the file you want to write
        file_format (str): must be one of 'csv' or 'feather'. feather is better for very large files


    Returns:
        None
    """

    #check inputs
    assert file_format in ['csv', 'feather']

    if file_format == 'csv':
        df_umap['embedding_dim_1'] = umap1
        df_umap['embedding_dim_2'] = umap2
        df_umap.to_csv(save_name, index=False)
        print('done.')

    elif file_format == 'feather':
        df_umap['embedding_dim_1'] = umap1
        df_umap['embedding_dim_2'] = umap2
        df_umap.to_feather(save_name)
        print('done.')

def spec_avg_from_list(spec_list):
    """
    Get the average spectrogram from a list of spectrograms

    Arguments:
        spec_list (list): a list of nonlinearized spectrograms as numpy arrays

    Returns:
        avg_spec_image (numpy array): the average of spectrograms in the list

    """

    #get average
    avg_spec_image = np.mean(spec_list, axis=0)

    return avg_spec_image

def show_specs(frame, num_freq_bins, num_time_bins, columns_to_drop):
    """
    Plot spectrogram images from a dataframe

    Arguments:
       frame (dataframe): a dataframe of pixel values where each row is a spectrogram and columns are pixel IDs (eg output of save_umap)
       num_freq_bins (int): number of frequency bins the spectrograms have
       num_time_bins (int): number of time bins the spectrograms have
       columns_to_drop (list of strings): columns that are not spectrogram pixel IDs

    Returns:
        None

    """

    for i in range(len(frame)):
        print(frame['source_file'].iloc[i])
        to_plot = frame.drop(columns = columns_to_drop)
        img = to_plot.iloc[i]
        img = np.array(img).reshape((num_freq_bins, num_time_bins))
        plt.imshow(img, origin = 'lower', extent = (num_freq_bins, 0, num_time_bins, 0 ))
        plt.show()

def files_from_umap(frame, umap1_name, umap2_name, umap1_thresh, umap2_thresh, source_dir):
    """
    Get the paths to the wav files in all or a portion of a umap embedding.
    Useful for getting spectrograms and/or spectrograms averages from particular regions of UMAP space.

    Arguments:
        frame (dataframe): a dataframe of pixel values where each row is a spectrogram and columns are pixel IDs
        umap1_name (str): the column name in frame for the umap x coordinate
        umap2_name (str): the column name in frame for the umap y coordinate
        umap1_thresh (list of two floats): list of minimum and maximum of UMAP x coordinate from which to get paths
        umap2_thresh (list of two floats): list of minimum and maximum of UMAP y coordinate from which to get paths
        source_dir (str): path to the directory containing all of the wav clips that went into the embedding

    Returns:
        source_files (list): list of paths to wall wav clips in the square defined by umap1_thresh and umap2_thresh

    """

    #get the spectrograms
    temp = frame[(frame[umap1_name] > umap1_thresh[0]) & (frame[umap1_name] < umap1_thresh[1]) & (frame[umap2_name] > umap2_thresh[0]) & (frame[umap2_name] < umap2_thresh[1])]

    #get the paths to their source files
    source_files = [source_dir+i for i in temp['source_file']]
    return source_files

def spec_avg_from_umap(frame, umap1_name, umap2_name, umap1_thresh, umap2_thresh, num_freq_bins, num_time_bins):
    """
    Get an average of all the spectrograms in a region of umap space

    Arguments:
        frame (dataframe): a dataframe of pixel values where each row is a spectrogram and columns are pixel IDs
        umap1_name (str): the column name in frame for the umap x coordinate
        umap2_name (str): the column name in frame for the umap y coordinate
        umap1_thresh (list of two floats): list of minimum and maximum of UMAP x coordinate from which to get paths
        umap2_thresh (list of two floats): list of minimum and maximum of UMAP y coordinate from which to get paths
        num_freq_bins (int): number of frequency bins used to generate the spectrogram
        num_time_bins (int): number of time bins used to generate the spectrogram

    Returns:
        source_files (list): list of paths to wall wav clips in the square defined by umap1_thresh and umap2_thresh

    """

    #get the spectrograms
    temp = frame[(frame[umap1_name] > umap1_thresh[0]) & (frame[umap1_name] < umap1_thresh[1]) & (frame[umap2_name] > umap2_thresh[0]) & (frame[umap2_name] < umap2_thresh[1])]

    #get the average
    pixels = [str(i) for i in range(num_freq_bins*num_time_bins)]
    avg_spec_image = np.mean(np.array(temp[pixels]), axis=0).reshape((num_freq_bins, num_time_bins))

    #return the individual spectrograms
    return temp, avg_spec_image

def specs_from_umap(frame, umap1_name, umap2_name, umap1_thresh, umap2_thresh, num_freq_bins, num_time_bins):
    """
    Show all the spectrograms in a region of umap space

    Arguments:
        frame (dataframe): a dataframe of pixel values where each row is a spectrogram and columns are pixel IDs
        umap1_name (str): the column name in frame for the umap x coordinate
        umap2_name (str): the column name in frame for the umap y coordinate
        umap1_thresh (list of two floats): list of minimum and maximum of UMAP x coordinate from which to get paths
        umap2_thresh (list of two floats): list of minimum and maximum of UMAP y coordinate from which to get paths
        num_freq_bins (int): number of frequency bins used to generate the spectrogram
        num_time_bins (int): number of time bins used to generate the spectrogram

    Returns:
       specs (list): a list of spectrogram images
       names (list): the names of the files that generate the images in specs

    """

    #get the spectrograms
    frame = frame[(frame[umap1_name] > umap1_thresh[0]) & (frame[umap1_name] < umap1_thresh[1]) & (frame[umap2_name] > umap2_thresh[0]) & (frame[umap2_name] < umap2_thresh[1])].copy()
    
    pixels = [str(i) for i in range(num_freq_bins*num_time_bins)]
        
    specs = []
    names = []
    for i in range(len(frame)):
        
        img = frame[pixels].iloc[i]
        img = np.array(img).reshape((num_freq_bins, num_time_bins))
        specs.append(img)
        names.append(frame['clip_names'].iloc[i])
        
    return specs, names

def ava_get_spec(audio, p):
    """
    From https://autoencoded-vocal-analysis.readthedocs.io/en/latest/_modules/ava/segmenting/utils.html?highlight=get_spec#
    Get a spectrogram. Much simpler than ``ava.preprocessing.utils.get_spec``.

    Arguments: 
       audio (Audio): numpy array of floats
       p (dict): Spectrogram parameters. Should the following keys: `'fs'`, `'nperseg'`,`'noverlap'`, `'min_freq'`, `'max_freq'`, `'spec_min_val'`,
        `'spec_max_val'`

    Returns:
        spec (numpy array of floats): Spectrogram of shape [freq_bins x time_bins]
        dt (float): Time step between time bins.
        f (numpy.ndarray): Array of frequencies.
    """

    #get log spectrograms between min_freq and max_fre1
    assert len(audio) >= p['nperseg'], "len(audio): " + str(len(audio)) + ", nperseg: " + str(p['nperseg'])
    f, t, spec = stft(audio, fs=p['fs'], nperseg=p['nperseg'], noverlap=p['noverlap'])
    i1 = np.searchsorted(f, p['min_freq'])
    i2 = np.searchsorted(f, p['max_freq'])
    f, spec = f[i1:i2], spec[i1:i2]
    spec = np.log(np.abs(spec))

    #apply thresholds and scale
    spec -= p['spec_min_val']
    spec /= p['spec_max_val'] - p['spec_min_val']
    spec = np.clip(spec, 0.0, 1.0)
    return spec, t[1]-t[0], f
