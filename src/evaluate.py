# this file contains functions for evaluating neural network trained to predict vocalizations

#filesystem
import os
import glob
from tqdm import tqdm
from pathlib import Path
#from das.utils import load_model_and_params

#plotting
import seaborn as sns 
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual, widgets

#data
import das
import time
import random
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import stft
from scipy.interpolate import interp2d
from sklearn.preprocessing import StandardScaler

import librosa
import librosa.display

from src.segmentation import das_predict, segments_to_dataframe
from src.parameters import load_json

def predict(path_to_wav, models_root, path_to_annotation, model_ID, thresholds, min_dur, min_intersyllable, new_das = False, save = False):
	"""
	Take an annotated recording, predict vocalizations from that recording using a range of thresholds

	Arguments:
		path_to_annotation (str): Full path to the annotations (each row a vocalization, columns for start and stop time)
		model_ID (str): model ID in the format yyyymmdd_hhmmss
		thresholds (list): list of float between 0 and 1 (eg [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) model will predict with 
		min_dur (float): minimum duration for vocalizations in seconds (predictions shorter than this will be dropped)
		min_intersyllable (float): minimum inter vocalization interval in seconds (vocs separated by shorter intervals will be merged)


	Returns:
		prediction_dfs (list): A list of dataframes where each dataframe contains predictions with a given threshold

	"""

	from das.utils import load_model_and_params

	#load the model and model parameters
	print('loading model...')
	deployment = path_to_wav.split('/')[-1].split('_')[1]
	model_path = os.path.join(models_root, model_ID)
	model, params = load_model_and_params(model_path)

	#Predict vocalizations using each threshold in thresholds
	prediction_dfs = []

	# set the predictions parameters you want to iterate over
	for segment_thresh in thresholds:
		print('predicting with threshold =', segment_thresh,'...')

		#predict
		_, segments, _, _ = das_predict(wav = path_to_wav, 
										model = model, 
										params = params, 
										verbose = 2, 
										segment_thres = segment_thresh,
										segment_minlen = min_dur, 
										segment_fillgap= min_intersyllable,
										pad = True
													 )

		#make it a dataframe
		segments = segments_to_dataframe(segments = segments, 
										 deployment = deployment, 
										 segment_thres = segment_thresh,
										 segment_minlen = min_dur, 
										 segment_fillgap = min_intersyllable, 
										 model_name = model_ID, 
										 wav=path_to_wav, 
										 new_das = new_das)

		#give the predictions interpretable names (cry, USV, or noise)
		segments['voc_type'] = ['USV' if i==1 else 'squeak' if i==2 else 'noise' for i in segments['label']]

		prediction_dfs.append(segments)

	print('done.')

	return prediction_dfs

    
    
def segments(predictions, tolerances, voc_type, path_to_annotation):
    """
    
    Take a dataframe of predictions and corresponding annotations. Use these to calculate precision, recall, and F1
    scores for a range of tolerances.

    Arguments:
        predictions (list of dataframes): a list of predictions dataframe(s), eg output of evaluate.predict() or segmentation.das_predict()
        tolerances (list of float): list of tolerances in seconds to evaluate on, predicted start/stop times within +/- a given tolerance       of the annotated start/stop will be counted as true positives.
        voc_type (str): vocalization type to evaluate on ('cry' or 'USV)
        path_to_annotation (str): Full path to the annotations (each row a vocalization, columns for start and stop time)
         
    Returns:
        

    """
    
    #check inputs
    assert voc_type in ['squeak', 'USV'], "voc_type must be either 'squeak' or 'USV'"
    assert os.path.exists(path_to_annotation)
    
    #get true_df
    true_df = pd.read_csv(path_to_annotation)
    true_df['name'] = ['USV' if i=='whistle' else 'cry' for i in  true_df['name']]
    true_df = true_df.loc[true_df['name'] == voc_type]
    
    
    # initialize lists to collect data from all tolerance levels
    thresholds_per_tolerance = []
    precisions_per_tolerance = []
    recalls_per_tolerance = []
    F1s_per_tolerance = []
    tolerances_per_tolerance = []
    
    for tolerance in tolerances:

        #initialize lists to collect data from each tolerance level
        tolerance_levels = []
        thresholds = []
        precisions = []
        recalls = []
        F1s = []
        
        for pred_df in predictions:
            
            #check that all vocalization offset times follow onset times
            assert np.min(pred_df['stop_seconds'] - pred_df['start_seconds']) > 0, "some predicted vocalizations have 0 or negative length"
            
            #check that predictions do not overlap
            assert np.min([pred_df['start_seconds'].iloc[i+1] - pred_df['stop_seconds'].iloc[i] for i in range(len(pred_df) -1)]) > 0, "some predicted vocalizations overlap"
            
            #get the predictions just for cries or USVs
            pred_df = pred_df.loc[pred_df['voc_type'] == voc_type]
            
            #get the predicted start and stop times for each vocalization of this type
            predicted_times = [[i, j] for i, j in zip(pred_df['start_seconds'], pred_df['stop_seconds'])]
            
            #get the annotated (true) start and stop times for each vocalization of this type
            true_times =  [[i, j] for i, j in zip(true_df['start_seconds'], true_df['stop_seconds'])]

            ## create dictionaries for true and predicted where the key is the position of the onset, and value is ( onset, offset )
            on_off_set_dict_predicted_times = {}
            for on_off_set_example in predicted_times:
                on_off_set_dict_predicted_times[ on_off_set_example[0] ]  = on_off_set_example

            on_off_set_dict_true_times = {}
            for on_off_set_example in true_times:
                on_off_set_dict_true_times[ on_off_set_example[0] ]  = on_off_set_example

            total_num_true_syllables = len(true_times)
            total_num_predicted_syllables = len(predicted_times)  


            num_match = 0

            #for each prediction
    
            for predicted_onset_time in list(on_off_set_dict_predicted_times.keys()):
                

                #get the corresponding offset
                predicted_offset_time = on_off_set_dict_predicted_times[predicted_onset_time][1]
                
                #is there a true start within the tolerance window of the prediction start?
                in_start_window = [i[0] for i in on_off_set_dict_true_times.values() if np.abs(i[0] - predicted_onset_time)<=tolerance]
                
                #for each true start in the tolerance window, is its corresponding stop in the tolerance window of the predicted stop    
                in_stop_window = []
                for true_start in in_start_window:
                    #get the corresponding offset
                    true_stop = on_off_set_dict_true_times[true_start][1]
                    
                    #test if it is also in the tolerance window of the predicted stop
                    if (predicted_offset_time - tolerance) <= true_stop <= (predicted_offset_time + tolerance):
                        num_match +=1

            # evaluate
            syllable_precision = num_match / (total_num_predicted_syllables +1e-12 )
            syllable_recall = num_match / (total_num_true_syllables +1e-12 )
            syllable_f1 = 2/( 1/(syllable_precision+1e-12) + 1/(syllable_recall+1e-12 )  )

            #collect evaluations for this threshold's predictions
            thresholds.append(pred_df['segment_threshold'].unique()[0])
            precisions.append(syllable_precision)
            recalls.append(syllable_recall)
            F1s.append(syllable_f1)
            tolerance_levels.append(tolerance)
        
        #collect evaluations for this tolerance level
        thresholds_per_tolerance.extend(thresholds)
        precisions_per_tolerance.extend(precisions)
        recalls_per_tolerance.extend(recalls)
        F1s_per_tolerance.extend(F1s)
        tolerances_per_tolerance.extend(tolerance_levels)
        
    #collect all evaluations intoa dataframe and return
    evaluation_df = pd.DataFrame()
    evaluation_df['tolerance'] = tolerances_per_tolerance
    evaluation_df['threshold'] = thresholds_per_tolerance
    evaluation_df['precision'] = precisions_per_tolerance
    evaluation_df['recall'] = recalls_per_tolerance
    evaluation_df['F1'] = F1s_per_tolerance
        
    return evaluation_df

def prediction_errors(predictions, voc_type, path_to_annotation, tolerance):
	"""

	Take a dataframe of predictions and corresponding annotations. Use these to calculate predictions errors for start and stop times. Modified
	from code written by Nianlong Gu.

	Arguments:
		predictions (list of dataframes): a list of predictions dataframe(s), eg output of evaluate.predict() or segmentation.das_predict()
		voc_type (str): vocalization type to evaluate on ('cry' or 'USV)
		path_to_annotation (str): Full path to the annotations (each row a vocalization, columns for start and stop time)

	Returns:


	"""

	#check inputs
	assert voc_type in ['squeak', 'USV'], "voc_type must be either 'squeak' or 'USV'"
	assert os.path.exists(path_to_annotation)

	#get true_df
	true_df = pd.read_csv(path_to_annotation)
	true_df['name'] = ['USV' if i=='whistle' else 'squeak' for i in  true_df['name']]
	true_df = true_df.loc[true_df['name'] == voc_type]

	true_positive_start_errors = []
	true_positive_stop_errors = []
	all_start_errors = []
	all_stop_errors = []
	thresholds = []

	for pred_df in predictions:

		#check that all vocalization offset times follow onset times
		assert np.min(pred_df['stop_seconds'] - pred_df['start_seconds']) > 0, "some predicted vocalizations have 0 or negative length"

		#check that predictions do not overlap
		assert np.min([pred_df['start_seconds'].iloc[i+1] - pred_df['stop_seconds'].iloc[i] for i in range(len(pred_df) -1)]) > 0, "some predicted vocalizations overlap"

		#get the predictions just for squeaks or USVs
		pred_df = pred_df.loc[pred_df['voc_type'] == voc_type]

		#get the predicted start and stop times for each vocalization of this type
		predicted_times = [[i, j] for i, j in zip(pred_df['start_seconds'], pred_df['stop_seconds'])]

		#get the annotated (true) start and stop times for each vocalization of this type
		true_times =  [[i, j] for i, j in zip(true_df['start_seconds'], true_df['stop_seconds'])]

		## create dictionaries for true and predicted where the key is the position of the onset, and value is ( onset, offset )
		on_off_set_dict_predicted_times = {}
		for on_off_set_example in predicted_times:
			on_off_set_dict_predicted_times[ on_off_set_example[0] ]  = on_off_set_example
			
		on_off_set_dict_durations = {}
		for on_off_set_example in predicted_times:
			on_off_set_dict_durations[ on_off_set_example[0] ]  = on_off_set_example[1] - on_off_set_example[0]

		on_off_set_dict_true_times = {}
		for on_off_set_example in true_times:
			on_off_set_dict_true_times[ on_off_set_example[0] ]  = on_off_set_example

		total_num_true_syllables = len(true_times)
		total_num_predicted_syllables = len(predicted_times)  

		predicted_starts  = [i[0] for i in predicted_times]
		predicted_stops  = [i[1] for i in predicted_times]

		#for each annotated vocalization
		start_idx=[]
		stop_idx=[]
		predicted_durations = []
		predicted_onsets = []
		predicted_offsets = []
		true_offsets = []
		
		for true_onset_time in list(on_off_set_dict_true_times.keys()):

			#get the corresponding offset
			true_offset_time = on_off_set_dict_true_times[true_onset_time][1]

			#find the time to closest predicted start
			min_abs_start_error = np.min([np.abs(i - true_onset_time) for i in predicted_starts])
			closest_start = [i for i in range(len(predicted_starts)) if np.abs(predicted_starts[i] - true_onset_time) == min_abs_start_error][0]

			if closest_start not in start_idx and (min_abs_start_error < tolerance):
				true_pos_start_error = true_onset_time - predicted_starts[closest_start]
			else:
				start_error = float('nan')

			#find the time to clostest predicted stop
			min_abs_stop_error = np.min([np.abs(i - true_offset_time) for i in predicted_stops])
			closest_stop = [i for i in range(len(predicted_stops)) if np.abs(predicted_stops[i] - true_offset_time) == min_abs_stop_error][0]
			predicted_duration = on_off_set_dict_durations[predicted_starts[closest_start]]
			
			
			predicted_onset = predicted_starts[closest_start]
			predicted_offset = predicted_stops[closest_stop]
			start_error = true_offset_time - predicted_stops[closest_stop]
			stop_error = true_onset_time - predicted_starts[closest_start]
			
			if closest_stop not in stop_idx and (min_abs_stop_error < tolerance): #both start and stop must be in threshold
				true_pos_stop_error = true_offset_time - predicted_stops[closest_stop]
				stop_idx.append(closest_stop)
				start_idx.append(closest_start) 
			else:
				stop_error = float('nan')

			true_positive_start_errors.append(true_pos_start_error)
			true_positive_stop_errors.append(true_pos_stop_error)
			all_start_errors.append(start_error)
			all_stop_errors.append(stop_error)
			
			predicted_durations.append(predicted_duration)
			thresholds.append(pred_df['segment_threshold'].unique()[0])
			predicted_onsets.append(predicted_onset)
			predicted_offsets.append(predicted_offset)
			true_offsets.append(true_offset_time)

	#collect all evaluations intoa dataframe and return
	errors_df = pd.DataFrame()
	errors_df['threshold'] = thresholds
	errors_df['true_positive_start_errors'] = true_positive_start_errors
	errors_df['true_positive_stop_errors'] = true_positive_stop_errors
	errors_df['all_start_errors'] = all_start_errors
	errors_df['all_stop_errors'] = all_stop_errors
	errors_df['predicted_duration'] = predicted_durations
	errors_df['true_onset'] = list(on_off_set_dict_true_times.keys())
	errors_df['true_offset'] = true_offsets
	errors_df['predicted_onset'] = predicted_onsets
	errors_df['predicted_offset'] = predicted_offsets
								   
	return errors_df

def plot(predicted, actual, wav_path, window_duration, step, with_spec, start_at, clip_seconds, voc_colors):
    """
    
    Show predictions, annotations, and the spectrogram stacked on top of one another with a horizontal scroll.
    Set 

    Arguments:
        
         
    Returns:
        

    """
    
    start_time, stop_time = clip_seconds[0], clip_seconds[1]
    test_actual = actual.loc[actual['start_seconds'] > start_time].loc[actual['stop_seconds'] < stop_time]
    predicted = predicted.loc[predicted['start_seconds'] > start_time].loc[predicted['stop_seconds'] < stop_time]

    #load the audio
    y, sr = librosa.load(wav_path, sr=192000)
    
    #clip it
    start = int(np.round(start_time*sr))
    stop = int(np.round(stop_time*sr))
    y = y[start:stop]

    #make the spectrogram
    n_fft = 128
    hop_length = 128//4
    D = librosa.stft(y, 
                 n_fft=n_fft, 
                 hop_length=hop_length, 
                 win_length=None, 
                 window='hann', 
                 center=True, 
                 dtype=None, 
                 pad_mode='constant')  

    specgram = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    #beginning and end of the plot
    #first_voc_start = np.min([list(actual['start_seconds'])[0], list(predicted['start_seconds'])[0]])
    #last_voc_end = np.max([list(actual['stop_seconds'])[-1], list(predicted['stop_seconds'])[-1]]) 
    
    first_voc_start = 0
    last_voc_end = 55

    def scroll_predictions(window_start):

        fig, axes = plt.subplots(nrows=3,
                                 ncols=1,
                                 figsize=[10,4], 
                                 dpi=300, 
                                 constrained_layout=True
                            )

        if with_spec:
            librosa.display.specshow(specgram, 
                                     hop_length=hop_length,
                                     cmap='viridis', 
                                     ax=axes[0],
                                     x_axis='s', 
                                     y_axis='linear', 
                                     sr=sr)
            


        Hz_values = axes[0].get_yticks() 
        axes[0].set_xticklabels(labels = [])
        axes[0].set_xticks([])
        axes[0].set_xlabel('')
        axes[0].set_yticks(ticks = np.arange(0, np.max(Hz_values), 20000))
        axes[0].set_yticklabels(labels = [int(i) for i in np.arange(0, np.max(Hz_values), 20000)/1000], fontsize=9)
        axes[0].set_ylabel('kHz')
        axes[1].set_xticklabels([])
        axes[1].set_xticks(ticks = np.arange(first_voc_start, last_voc_end, 1))
        axes[1].set_xticklabels(labels = [])
        axes[1].set_ylabel('true', rotation = 90)
        axes[2].set_ylabel('pred')
        

        #voc type names
        voc_types = ['squeak', 'USV']
        box_height = 1
        for voc_type in voc_types:

            true_starts = np.array(actual['start_seconds'].loc[actual['name'] == voc_type])
            true_stops = np.array(actual['stop_seconds'].loc[actual['name'] == voc_type])

            predicted_starts = np.array(predicted['start_seconds'].loc[predicted['voc_type'] == voc_type])
            predicted_stops = np.array(predicted['stop_seconds'].loc[predicted['voc_type'] == voc_type])

            for start, stop in zip(true_starts, true_stops):

                box_length = stop - start
                box = plt.Rectangle((start,0), box_length, box_height, alpha = 0.7, color = voc_colors[voc_type])
                axes[1].add_patch(box)

            for start, stop in zip(predicted_starts, predicted_stops):

                box_length = stop - start
                box = plt.Rectangle((start,0), box_length, box_height, alpha = 0.7, color = voc_colors[voc_type])
                axes[2].add_patch(box)

        sns.despine()

        axes[2].set_xlabel('time (s)')
        axes[2].set_xticks(ticks = np.arange(clip_seconds[0], clip_seconds[1], 0.1))
        axes[2].set_xticklabels(labels = np.round(np.arange(clip_seconds[0], clip_seconds[1], 0.1),1), fontsize=9)
        
        for ax, _ in enumerate(axes):
            if ax !=0:
                axes[ax].spines['left'].set_visible(False)
                axes[ax].set_yticks([])
                axes[ax].set_xlim([window_start, window_start+window_duration])
            elif ax == 0:
                axes[ax].set_xlim([window_start - clip_seconds[0], window_start+window_duration-clip_seconds[0]])
        
        

        plt.show()

    interact(scroll_predictions, 
             window_start = widgets.FloatSlider(value=start_at,
                                          min=clip_seconds[0],
                                          max=clip_seconds[1]-window_duration,
                                          step=step))
def make_dataframe(res, model_ID):
    """
    Make a dataframe from a das results file
    
    
    """
    
    
    chunk_eval = pd.DataFrame()

    F1_scores = [ res['classification_report']['cry']['f1-score'], res['classification_report']['USV']['f1-score'], res['classification_report']['noise']['f1-score'] ]
    precisions = [ res['classification_report']['cry']['precision'], res['classification_report']['USV']['precision'], res['classification_report']['noise']['precision'] ]
    recalls = [ res['classification_report']['cry']['recall'], res['classification_report']['USV']['recall'], res['classification_report']['noise']['recall'] ]        
    voc_types = ['cry', 'USV', 'noise']
    model = ['v2']*3

    chunk_eval['F1_score'] = F1_scores
    chunk_eval['precision'] = precisions
    chunk_eval['recall'] = recalls
    chunk_eval['voc_type'] = voc_types
    chunk_eval['model'] = model_name
           
		
def calculate_metrics(comparison_df):
	"""
	Evaluate model counts
	"""

	# Binary classification of predictions and actuals
	comparison_df['squeaks_predicted'] = [1 if i !=0 else 0 for i in comparison_df['predicted_squeak_count']] 
	comparison_df['USV_predicted'] = [1 if i !=0 else 0 for i in comparison_df['predicted_USV_count']] 
	comparison_df['voc_predicted'] = [1 if i+j !=0 else 0 for i,j in zip(comparison_df['predicted_USV_count'], comparison_df['predicted_squeak_count'])] 
	comparison_df['any_actual_squeaks'] = [1 if i !=0 else 0 for i in comparison_df['actual_squeak_count']]
	comparison_df['any_actual_USV'] = [1 if i !=0 else 0 for i in comparison_df['actual_USV_count']]

	# Calculate differences for prediction correctness
	comparison_df['squeak_prediction_diff'] = comparison_df['any_actual_squeaks'] - comparison_df['squeaks_predicted']
	comparison_df['USV_prediction_diff'] = comparison_df['any_actual_USV'] - comparison_df['USV_predicted']
	comparison_df['voc_prediction_diff'] = comparison_df['vocs?'] - comparison_df['voc_predicted']

	# Confusion Matrix Calculation
	squeaks_predicted_squeaks = len([i for i,j in zip(comparison_df['squeak_prediction_diff'], comparison_df['any_actual_squeaks']) if i+j == 1])
	USV_predicted_USVs = len([i for i,j in zip(comparison_df['USV_prediction_diff'], comparison_df['any_actual_USV']) if i+j == 1])
	voc_predicted_vocs = len([i for i,j in zip(comparison_df['voc_prediction_diff'], comparison_df['vocs?']) if i+j == 1])

	no_squeak_predicted_no_squeak = len([i for i,j in zip(comparison_df['squeak_prediction_diff'], comparison_df['any_actual_squeaks']) if i+j == 0])
	no_USV_predicted_no_USV = len([i for i,j in zip(comparison_df['USV_prediction_diff'], comparison_df['any_actual_USV']) if i+j == 0])
	no_voc_predicted_no_voc = len([i for i,j in zip(comparison_df['voc_prediction_diff'], comparison_df['vocs?']) if i+j == 0])

	squeak_predicted_no_squeak = len([i for i in comparison_df['squeak_prediction_diff'] if i == 1])
	USV_predicted_no_USV = len([i for i in comparison_df['USV_prediction_diff'] if i == 1])
	voc_predicted_no_voc = len([i for i in comparison_df['voc_prediction_diff'] if i == 1])

	no_squeak_predicted_squeak = len([i for i in comparison_df['squeak_prediction_diff'] if i == -1])
	no_USV_predicted_USV = len([i for i in comparison_df['USV_prediction_diff'] if i == -1])
	no_voc_predicted_voc = len([i for i in comparison_df['voc_prediction_diff'] if i == -1])

	# Confusion Matrices
	squeak_presence_matrix = np.array([[squeaks_predicted_squeaks, no_squeak_predicted_squeak], [squeak_predicted_no_squeak, no_squeak_predicted_no_squeak]])
	USV_presence_matrix = np.array([[USV_predicted_USVs, no_USV_predicted_USV], [USV_predicted_no_USV, no_USV_predicted_no_USV]])
	voc_presence_matrix = np.array([[voc_predicted_vocs, no_voc_predicted_voc], [voc_predicted_no_voc, no_voc_predicted_no_voc]])

	# Precision, Recall, and F1 Score Calculation
	squeak_precision = squeaks_predicted_squeaks / comparison_df['squeaks_predicted'].sum()
	squeak_recall = squeaks_predicted_squeaks / comparison_df['any_actual_squeaks'].sum()
	squeak_F1 = 2 * (squeak_precision * squeak_recall) / (squeak_precision + squeak_recall)

	USV_precision = USV_predicted_USVs / comparison_df['USV_predicted'].sum()
	USV_recall = USV_predicted_USVs / comparison_df['any_actual_USV'].sum()
	USV_F1 = 2 * (USV_precision * USV_recall) / (USV_precision + USV_recall)

	vocs_precision = voc_predicted_vocs / comparison_df['voc_predicted'].sum()
	vocs_recall = voc_predicted_vocs / comparison_df['vocs?'].sum()
	vocs_F1 = 2 * (vocs_precision * vocs_recall) / (vocs_precision + vocs_recall)

	metrics_dict = {'squeak': {'precision': squeak_precision, 'recall': squeak_recall, 'F1': squeak_F1},
					'USV': {'precision': USV_precision, 'recall': USV_recall, 'F1': USV_F1},
					'vocs': {'precision': vocs_precision, 'recall': vocs_recall, 'F1': vocs_F1}}


	return metrics_dict, squeak_presence_matrix, USV_presence_matrix, voc_presence_matrix

def plot_confusion_matrix(ax, data, title, xlabel, ylabel, cbar, labels, vmin=0, vmax=1):
	"""
	Plot an annotated confusion matrix
	"""
	cax = ax.imshow(data, cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
	if cbar:
		cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
		cbar.outline.set_visible(False)
	ax.set_xticks(np.arange(len(labels)))
	ax.set_yticks(np.arange(len(labels)))
	ax.set_xticklabels(labels)
	ax.set_yticklabels(labels)
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	for i in range(len(labels)):
		for j in range(len(labels)):
			text = ax.text(j, i, data[i, j], ha="center", va="center", color="w", fontsize=10)
	ax.set_title(title)
	ax.set_xlabel(xlabel)
	if ylabel:
		ax.set_ylabel(ylabel)
	for spine in ax.spines.values():
		spine.set_visible(False)


           
    
    
    
    