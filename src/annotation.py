#this file contains functions for keeping track of annotated recordings and vocalizations within those recordings
import os
import numpy as np
from glob import glob
import pandas as pd
from pathlib import Path
from datetime import timedelta

from src.timestamps import get_deployments, audiomoth_to_datetime, get_audiomoth_from_box, time_correction
from src.rfid import fk_id_to_box_id
from src.parameters import load_json

import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_annotated(annotations_root):
    """
    get a list of wav files that have been annottaed from a particular annotation iteration
    iteration is the path to the directory containing the set of annotations you want to list
    """
    
    #this is teh directory with annotations for getting segment F1 scores
    segments_test_dir = os.path.join(annotations_root, 'segments_test')

    annotations = [i for i in os.listdir(iteration) if i.endswith('.csv')]
    segments_test = [i for i in os.listdir(segments_test_dir) if i.endswith('.csv')]
    
    already_annotated = annotations + segments_test
    
    return already_annotated
    
    
def check_annotated(wav_path, iteration):
    """
    check if a wav clip from a given audiomoth has been annotated and return True if so (False if not)
    iteration is the path to the directory containing the set of annotations in which you want to check
    wav is the full path to the wav in the raw audio directory
    """
    
    #check inputs make sense
    assert os.path.exists(wav_path)
    assert wav_path.split('/')[-3] in ['audiomoth00', 'audiomoth01', 'audiomoth02', 'audiomoth03', 'audiomoth04']
    assert wav_path.split('/')[-1].endswith('.wav')
    
    #get the info
    moth = wav_path.split('/')[-3][5:]
    box = wav_path.split('/')[-2].split('_')[-1]
    wav = Path(wav_path).stem
    date = wav.split('_')[0]
    time = wav.split('_')[-1]
    
    #convert to annotation naming format
    annotation_name = ('_').join([date, box, moth, time, 'annotations.csv'])
    
    return annotation_name in get_annotated(iteration)
    
    
def count_annotated(iteration):
	"""
	get how many (number and total duration in seconds) cries and USVs have been annotated
	iteration is the path to the directory containing the set of annotations you want to count
	"""

	#get a list of each annotation csv
	annotations = glob(iteration+'/*.csv')

	files, boxes, moths, cry_counts, cry_durations, USV_counts, USV_durations = [], [], [], [], [], [], []
	for annotation in annotations:
		
		source_file = Path(annotation).stem

		#get the data
		df = pd.read_csv(annotation)

		#check that the annotations are cry or USV
		assert set(df['name'].unique()) == set(['cry', 'USV']), "You have annotations that are not cry or USV"

		#check that all start and stop times are positive
		assert (np.nanmin(df['start_seconds'])) >= 0 or (np.isnan(np.nanmin(df['start_seconds']))), "Some annotated start times are negative"
		assert (np.nanmin(df['stop_seconds'])) >= 0 or (np.isnan(np.nanmin(df['start_seconds']))), "Some annotated stop times are negative"

		#get the durations
		df['duration'] = df['stop_seconds'] - df['start_seconds']

		#check all the stop times come after start times
		assert (np.nanmin(df['duration']) >= 0) or (np.isnan(np.nanmin(df['duration']))), "Some start times come after corresponding stop times"

		#check for unusually long vocalizations suggesting bad annotation
		if np.nanmax(df['duration'].loc[df['name'] == 'cry']) > 2:
			print('Warning! There are annotated cries longer than 2 seconds')
		if np.nanmax(df['duration'].loc[df['name'] == 'USV']) > 1:
			print('Warning! There are annotated USV longer than 1 seconds')
	        
		
		box = source_file.split('_')[1]
		moth = source_file.split('_')[2]

		cry_duration = df['duration'].loc[df['name'] == 'cry'].sum()
		USV_duration = df['duration'].loc[df['name'] == 'USV'].sum()

		if cry_duration != 0:
			cry_count = len(df.loc[df['name'] == 'cry'])
		else: 
			cry_count = 0

		if USV_duration != 0:
			USV_count = len(df.loc[(df['name'] == 'whistle') | (df['name'] == 'USV')])
		else:
			USV_count = 0

		#add it to the list
		files.append(source_file)
		boxes.append(box)
		moths.append(moth)
		cry_counts.append(cry_count)
		USV_counts.append(USV_count)
		cry_durations.append(cry_duration)
		USV_durations.append(USV_duration)


	annotations_df = pd.DataFrame()
	annotations_df['moth'] = moths
	annotations_df['box'] = boxes
	annotations_df['source_file'] = files
	annotations_df['cry_count'] = cry_counts
	annotations_df['USV_count'] = USV_counts
	annotations_df['cry_duration'] = cry_durations
	annotations_df['USV_duration'] = USV_durations
	annotations_df['annotation_iteration'] = iteration

	return annotations_df

def check_chime_annotations(root, chimes_dir, verbose = False):
    """
    check annotation files all have the correct naming convention
    return a list of deployments with at least one set of chime annotations (ie at least at dropoff)
    """
    deployments = get_deployments(root)
    deployment_info_wrong = []
    not_annotated = []
    missing_info = []
    info_in_wrong_format = []
    missing_deployment_chime = []
    missing_recovery_chime = []
    missing_npz = []
    
    #no annotation possible (bc no chime was used)
    no_chimes = ["20220610-20220612", 
                  "20220617-20220619",
                  "20220621-20220623",
                  "20220623-20220624",
                  "20220624-20220626",
                  "20220706-20220708",
                  "20220713-20220715",
                  "20220715-20220717",
                  "20220718-20220720",
                  "20220722-20220724",
                  "20220812-20220814",
                  "20220817-20220819",
                  "20220916-20220918"]
        
    deployments = [i for i in deployments if i not in no_chimes]
    
    #check that all the files start with a deployment
    annotations_files = [i for i in os.listdir(chimes_dir) if i.endswith('.csv') and not i.startswith('.')]
    file_name_starts = [i.split('_')[0] for i in annotations_files]
    for file in annotations_files:
        if file.split('_')[0] not in deployments:
            deployment_info_wrong.append(file)
            
    #for each deployment
    for deployment in deployments:

        #get the annotations available
        chimes_files = [os.path.join(chimes_dir, i) for i in os.listdir(chimes_dir) if i.startswith(deployment) and i.endswith('.csv')]

        #check if there is a chime annotation
        if (len(chimes_files) == 0) and (not deployment in no_chimes):
            not_annotated.append(deployment)
            
        else:
            
            for file_path in chimes_files:
                
                name = file_path.split('/')[-1]
                
                #check that each file has 4 underscores
                if len(name.split('_')) != 5:
                    missing_info.append(name)
                    
                else:
                    
                    #check that file has expected format
                    if not name.split('_')[1].startswith('box'):
                        info_in_wrong_format.append(name)
                        
                    elif not name.split('_')[1].split('box')[-1].isdigit():
                        info_in_wrong_format.append(name)
                        
                    elif not (name.split('_')[2].isdigit() and len(name.split('_')[2]) == 6):
                        info_in_wrong_format.append(name)
                        
                    elif not name.split('_')[3] in ['deployment', 'recovery']:
                        info_in_wrong_format.append(name)
                        
                    elif not name.split('_')[4] == 'annotations.csv':
                        info_in_wrong_format.append(name)
                    
                    #check that there is a corresponding npz
                    npz_name = name.split('_annotation')[0]+'.npz'
                    if npz_name not in os.listdir(chimes_dir):
                        missing_npz.append(name)
                        
            
                
    if verbose:        
        #show info
        print('The following deployments do not have chime annotations:')
        for i in not_annotated:
            print('\t', i  )

        print('The following file names have incorrectly written deployments:')
        for i in deployment_info_wrong:
            print('\t', i  )

        print('The following file names are missing information:')
        for i in missing_info:
            print('\t', i  )    

        print('The following file names have information in the wrong format:')
        for i in info_in_wrong_format:
            print('\t', i  )    

        print('The following files do not have an accompanying npz file with the correct naming convention:')
        for i in missing_npz:
            print('\t', i  )    
    
    annotated_deployments = [i for i in deployments if i not in not_annotated]
    return annotated_deployments
                
                    

    
def collect_chime_annotations(root, deployment, chimes_dir, save_dir):
	"""
	collect all the annotations for a single deplyment into a df and save so that there
	is a 1:1 correspondence bwteen each csv in 
	/Volumes/LaCie_barn/mouse_barn_audiomoth/data/rfid/test_transponder/audiomoth_readings
	and 
	/Volumes/LaCie_barn/mouse_barn_audiomoth/data/rfid/test_transponder/rfid_readings


	comes from collect chime annotations in Annotate Chimes_old.ipynb
	"""

	if not os.path.exists(os.path.join(save_dir, ('_').join([deployment, 'audiomoth_chime_readings.csv']))):

		active_fk_box_ids = [79, 74, 84, 73, 67, 62, 57, 53, 69, 52, 48, 63, 59, 50, 85, 47, 87, 83, 66, 68]

		#get the chimes files
		chimes_files = [os.path.join(chimes_dir, i) for i in os.listdir(chimes_dir) if i.startswith(deployment) and i.endswith('.csv')]

		#add source file and some other info to chimes
		for chime_file in chimes_files:
			df = pd.read_csv(chime_file)
			if not 'source_file' in df.columns:
				df['source_file'] = os.path.split(chime_file)[-1]
				df.to_csv(chime_file, index = False)

		chimes_df = pd.concat([pd.read_csv(i) for i in chimes_files]).reset_index(drop=True)
		chimes_df['box'] = [int(i.split('_')[-4].split('box')[-1]) for i in chimes_df['source_file']]
		chimes_df['start_minute'] = [int(i.split('_')[-3]) for i in chimes_df['source_file']]
		chimes_df['recovery'] = [1 if i.split('_')[-2] == 'recovery' else 0 for i in chimes_df['source_file']]

		#convert audiomoth time to datetime time
		chimes_df['start_time'] = [('_').join([deployment.split('-')[0], str(i)]) if j == 0 else ('_').join([deployment.split('-')[1], str(i)]) for i,j in zip(chimes_df['start_minute'], chimes_df['recovery'])]
		chimes_df['start_time'] = [audiomoth_to_datetime(i) for i in chimes_df['start_time']]
		start_time = []

		for i, secs in zip(chimes_df['start_time'], chimes_df['start_seconds']):
			start_time.append(i+timedelta(seconds=secs))

		chimes_df['event_time']  = start_time #the onset of the chime should correspond to event_time in the test transponder rfid table
		chimes_df.to_csv(os.path.join(save_dir, ('_').join([deployment, 'audiomoth_chime_readings.csv'])), index=False)

	else:
		print(os.path.join(save_dir, ('_').join([deployment, 'audiomoth_chime_readings.csv']), 'already exists...'))

def make_time_corrections_df(root, audiomoths, verbose = False):
	"""
	Make a df where each row is a box that has both a deployment and recovery annotation and chime, 
	and columns are timedelta info needed for audiomoth timestamp correction

	"""

	#path to json listing the deployments for which no chimes were generated (therefore no time correction possible)
	no_chimes_path = os.path.join(root, 'parameters', 'no_chimes_generated.json')

	#path to json listing the boxes recorded fro each audiomoth on each deployment
	boxes_recorded_path = os.path.join(root, 'parameters', 'boxes_recorded.json')

	#path to directory containing aggregated chime annotations for each deployment
	chimes_root = os.path.join(root, 'data', 'rfid', 'test_transponder', 'audiomoth_readings')

	# path to the directory containing the raw chime annotations
	chimes_dir = os.path.join(root, 'data', 'annotations', 'chimes')

	#path to the directory containing the rfid test transponder readings that produced annotated chimes
	rfid_root = os.path.join(root,'data', 'rfid', 'test_transponder', 'rfid_readings')

	#get the deployments that have chime annotations

	deployments_list = [] 
	boxes_list = [] 
	audiomoths_list = []
	first_deployment_chimes = []
	closest_deployment_rfids = []
	deployment_corrections = []
	first_recovery_chimes = []
	closest_recovery_rfids = []
	recovery_corrections = []
	rfid_total_times = []
	chime_total_times = []
	total_time_errors = []
	actual_or_estimated = []

	annotated_deployments = check_chime_annotations(root, chimes_dir, verbose = False)

	for deployment in annotated_deployments:

		if verbose:
			print("Making time corrections df for:", deployment)

		#get the boxes recorded on this deployment
		boxes_recorded = load_json(boxes_recorded_path)
		boxes_recorded = [boxes_recorded[moth][deployment] for moth in audiomoths if deployment in boxes_recorded[moth].keys()]

		#find the chime annotation csv
		chime_df_path = os.path.join(chimes_root, deployment+'_audiomoth_chime_readings.csv')
		chime_df = pd.read_csv(chime_df_path)

		#find the rfid test transponder csv
		test_df_path = os.path.join(rfid_root, deployment+'_test_transponder_readings.csv')
		test_df = pd.read_csv(test_df_path)

		#ignore test transponder readings except at recorded boxes
		test_df = test_df[test_df['box'].isin(boxes_recorded)]

		if set(test_df['box'].unique()) == set(chime_df['box'].unique()):
			chime_df_boxes = set(chime_df['box'].unique())
			test_df = test_df[test_df['box'].isin(chime_df_boxes)] # only consider boxes you recorded from
		boxes = list(set(test_df['box'].unique()))

		#for each box
		for box in boxes:

			#get the audiomoth
			audiomoth = get_audiomoth_from_box(root = root, box=box, deployment=deployment)

			#subset the dfs by box
			chime_box = chime_df[chime_df['box'] == box]
			test_box = test_df[test_df['box'] == box]

			#if there are deployment and recovery annotations for both
			if set(chime_box['recovery']) == set([0,1]): 

				#get the box and deployment for the df
				audiomoths_list.append(audiomoth)
				boxes_list.append(box)
				deployments_list.append(deployment)

				#get the first deployment chime 
				first_deployment_chime = pd.to_datetime(chime_box['event_time'])[chime_box['recovery'] == 0].min()
				first_deployment_chimes.append(first_deployment_chime)

				#find the closest rfid read
				idx = pd.to_datetime(test_box['event_time'][test_box['recovery'] == 0]).sub(first_deployment_chime).abs().idxmin()
				closest_deployment_rfid = pd.to_datetime(test_box.loc[idx, 'event_time'])
				closest_deployment_rfids.append(closest_deployment_rfid)

				#get the difference (how much should you add to first_deployment_chime to make it = to  closest_rfid_read)
				deployment_correction =  first_deployment_chime - closest_deployment_rfid
				deployment_corrections.append(deployment_correction)

				#do the same for recovery
				#get the first recovery chime 
				first_recovery_chime = pd.to_datetime(chime_box['event_time'])[chime_box['recovery'] == 1].min()
				first_recovery_chimes.append(first_recovery_chime)

				#find the closest rfid read
				idx = pd.to_datetime(test_box['event_time'][test_box['recovery'] == 1]).sub(first_deployment_chime).abs().idxmin()
				closest_recovery_rfid = pd.to_datetime(test_box.loc[idx, 'event_time'])
				closest_recovery_rfids.append(closest_recovery_rfid)

				#get the difference (how much should you add to first_deployment_chime to make it = to  closest_rfid_read)
				recovery_correction =  first_recovery_chime - closest_recovery_rfid
				recovery_corrections.append(recovery_correction)

				#make sure everything adds up
				assert closest_deployment_rfid + deployment_correction == first_deployment_chime
				assert closest_recovery_rfid + recovery_correction == first_recovery_chime

				#get how much time passed between the first deployment rfid and the first recovery rfid signal
				rfid_total_time = closest_recovery_rfid - closest_deployment_rfid
				rfid_total_time = rfid_total_time.total_seconds()
				rfid_total_times.append(rfid_total_time)

				#get how much time passed between the first deployment chime and the first recovery rfid chime
				chime_total_time = first_recovery_chime - first_deployment_chime
				chime_total_time = chime_total_time.total_seconds()
				chime_total_times.append(chime_total_time)

				#get the difference (error in total duration)
				total_time_error = chime_total_time - rfid_total_time
				total_time_errors.append(total_time_error)

				#indicate that this correction is from an actual chime
				actual_or_estimated.append('actual')

	#collect the data in a dataframe
	time_corrections_df = pd.DataFrame()
	time_corrections_df['deployment'] = deployments_list
	time_corrections_df['moth'] = audiomoths_list
	time_corrections_df['box'] = boxes_list
	time_corrections_df['first_deployment_chime'] = first_deployment_chimes 
	time_corrections_df['closest_deployment_rfid'] = closest_deployment_rfids 
	time_corrections_df['deployment_correction'] = deployment_corrections 
	time_corrections_df['first_recovery_chime'] = first_recovery_chimes 
	time_corrections_df['closest_recovery_rfid'] = closest_recovery_rfids 
	time_corrections_df['recovery_correction'] = recovery_corrections 
	time_corrections_df['recording_duration_on_rfid_clock'] = rfid_total_times
	time_corrections_df['recording_duration_on_audiomoth_clock'] = chime_total_times
	time_corrections_df['recording_duration_error_seconds'] = total_time_errors
	time_corrections_df['deployment_correction_seconds'] = [i.total_seconds() for i in time_corrections_df['deployment_correction']]
	time_corrections_df['recovery_correction_seconds'] = [i.total_seconds() for i in time_corrections_df['recovery_correction']]
	time_corrections_df['atual_or_estimated'] = actual_or_estimated

	return time_corrections_df

def make_time_correction_estimates_df(root, audiomoths, verbose = False):
	"""
	Make a df where each row is a box that has both a deployment chime and recovery chime, 
	and columns are timedelta info needed for audiomoth timestamp correction, using average recovery chime 
	from this moth on other deployments to estimate clock drift

	"""

	#path to json listing the deployments for which no chimes were generated (therefore no time correction possible)
	no_chimes_path = os.path.join(root, 'parameters', 'no_chimes_generated.json')

	#path to json listing the boxes recorded fro each audiomoth on each deployment
	boxes_recorded_path = os.path.join(root, 'parameters', 'boxes_recorded.json')

	#path to directory containing aggregated chime annotations for each deployment
	chimes_root = os.path.join(root, 'data', 'rfid', 'test_transponder', 'audiomoth_readings')

	# path to the directory containing the raw chime annotations
	chimes_dir = os.path.join(root, 'data', 'annotations', 'chimes')

	#path to the directory containing the rfid test transponder readings that produced annotated chimes
	rfid_root = os.path.join(root,'data', 'rfid', 'test_transponder', 'rfid_readings')

	#get the deployments that have chime annotations

	deployments_list = [] 
	boxes_list = [] 
	audiomoths_list = []
	first_deployment_chimes = []
	closest_deployment_rfids = []
	deployment_corrections = []
	first_recovery_chimes = []
	closest_recovery_rfids = []
	recovery_corrections = []
	rfid_total_times = []
	chime_total_times = []
	total_time_errors = []
	actual_or_estimated = []

	#get the annotated deployments
	annotated_deployments = check_chime_annotations(root, chimes_dir, verbose = False)

	#get the annotated chimes and offsets and group recovery offset by moth and average
	actual_offsets = make_time_corrections_df(root, audiomoths = audiomoths, verbose = verbose)
	moth_average_offsets = actual_offsets[['recovery_correction_seconds', 'recording_duration_on_audiomoth_clock', 'moth']].groupby('moth').mean()

	for deployment in annotated_deployments:

		if verbose:
			print("Making estimated time corrections df for:", deployment)

		#get the boxes recorded on this deployment
		boxes_recorded = load_json(boxes_recorded_path)
		boxes_recorded = [boxes_recorded[moth][deployment] for moth in audiomoths if deployment in boxes_recorded[moth].keys()]

		#find the chime annotation csv
		chime_df_path = os.path.join(chimes_root, deployment+'_audiomoth_chime_readings.csv')
		chime_df = pd.read_csv(chime_df_path)

		#find the rfid test transponder csv
		test_df_path = os.path.join(rfid_root, deployment+'_test_transponder_readings.csv')
		test_df = pd.read_csv(test_df_path)

		#ignore test transponder readings except at recorded boxes
		test_df = test_df[test_df['box'].isin(boxes_recorded)]

		if set(test_df['box'].unique()) == set(chime_df['box'].unique()):
			chime_df_boxes = set(chime_df['box'].unique())
			test_df = test_df[test_df['box'].isin(chime_df_boxes)] # only consider boxes you recorded from
								 
			
		boxes = list(set(test_df['box'].unique()))

		#for each box
		for box in boxes:

			#get the audiomoth
			audiomoth = get_audiomoth_from_box(root = root, box=box, deployment=deployment)

			#subset the dfs by box
			chime_box = chime_df[chime_df['box'] == box]
			test_box = test_df[test_df['box'] == box]

			#if there are deployment and recovery annotations for both
			if set(chime_box['recovery']) == set([0]): 

				#get the box and deployment for the df
				audiomoths_list.append(audiomoth)
				boxes_list.append(box)
				deployments_list.append(deployment)

				#get the first deployment chime 
				first_deployment_chime = pd.to_datetime(chime_box['event_time'])[chime_box['recovery'] == 0].min()
				first_deployment_chimes.append(first_deployment_chime)

				#find the closest rfid read
				idx = pd.to_datetime(test_box['event_time'][test_box['recovery'] == 0]).sub(first_deployment_chime).abs().idxmin()
				closest_deployment_rfid = pd.to_datetime(test_box.loc[idx, 'event_time'])
				closest_deployment_rfids.append(closest_deployment_rfid)

				#get the difference (how much should you add to first_deployment_chime to make it = to  closest_rfid_read)
				deployment_correction =  first_deployment_chime - closest_deployment_rfid
				deployment_corrections.append(deployment_correction)

				#get the first recovery chime  - NaN because no recovery chime
				first_recovery_chimes.append(float('NaN'))

				#find the closest rfid read - NaN because no recovery chime
				closest_recovery_rfids.append(float('NaN'))

				#do the same for recovery, using the average offset for this moth in other deployments as an estimate of clock drift
				estimated_recovery_correction =  moth_average_offsets['recovery_correction_seconds'].loc[audiomoth]
				recovery_corrections.append(timedelta(days = 0, hours = 0, minutes = 0, seconds = estimated_recovery_correction))

				#make sure everything adds up
				assert closest_deployment_rfid + deployment_correction == first_deployment_chime

				#get how much time passed between the first deployment rfid and the first recovery rfid signal - NaN since no recovery chime
				rfid_total_times.append(float('NaN'))

				#get how much time passed between the first deployment chime and the first recovery rfid chime - NaN since no recovery chime
				estimated_duration =  moth_average_offsets['recording_duration_on_audiomoth_clock'].loc[audiomoth]
				chime_total_times.append(estimated_duration)

				#get the difference (error in total duration) - NaN since no recovery chime
				total_time_errors.append(float('NaN'))

				#indicate that this correction is from an actual chime
				actual_or_estimated.append('estimated')

	#collect the data in a dataframe
	time_correction_estimates_df = pd.DataFrame()
	time_correction_estimates_df['deployment'] = deployments_list
	time_correction_estimates_df['moth'] = audiomoths_list
	time_correction_estimates_df['box'] = boxes_list
	time_correction_estimates_df['first_deployment_chime'] = first_deployment_chimes 
	time_correction_estimates_df['closest_deployment_rfid'] = closest_deployment_rfids 
	time_correction_estimates_df['deployment_correction'] = deployment_corrections 
	time_correction_estimates_df['first_recovery_chime'] = first_recovery_chimes 
	time_correction_estimates_df['closest_recovery_rfid'] = closest_recovery_rfids 
	time_correction_estimates_df['recovery_correction'] = recovery_corrections 
	time_correction_estimates_df['recording_duration_on_rfid_clock'] = rfid_total_times
	time_correction_estimates_df['recording_duration_on_audiomoth_clock'] = chime_total_times
	time_correction_estimates_df['recording_duration_error_seconds'] = total_time_errors
	time_correction_estimates_df['deployment_correction_seconds'] = [i.total_seconds() for i in time_correction_estimates_df['deployment_correction']]
	time_correction_estimates_df['recovery_correction_seconds'] = [i.total_seconds() for i in time_correction_estimates_df['recovery_correction']]
	time_correction_estimates_df['atual_or_estimated'] = actual_or_estimated

	return time_correction_estimates_df

def correct_audiomoth_times(root, time_corrections_df, model_ID, save = True):
	"""
	Give a time corrections df and path to the project root, get an updated vocal events table for each recorded box
	with corrected time stamps. model_ID is the das model ID
	"""

	updated_vocal_events = []
	vocal_events_root = os.path.join(root, 'data', 'segments', 'vocal_events', model_ID)
	deployments = set([i.split('_')[0] for i in os.listdir(vocal_events_root) if not i.startswith('.')])

	for deployment in deployments:
		print(deployment)

		#get the boxes for this deplpyment
		boxes = time_corrections_df['box'][time_corrections_df['deployment'] == deployment]

		#for each box
		for box in boxes:
			print('\t', box)
			audiomoth = get_audiomoth_from_box(root, box, deployment)

			if audiomoth != 'audiomoth00':

				file_to_save = os.path.join(vocal_events_root, ('_').join([deployment,'box'+str(box),'_timeadjusted.csv']))

				if not os.path.exists(file_to_save):
					
					v_events_file = os.path.join(vocal_events_root, ('_').join([deployment, 'box' + str(box), 'segments.csv']))
					
					if not os.path.exists(v_events_file):
						print('Deployment', deployment, 'and box', box, 'have chime annotations but no vocal events file...')
						continue
					
					# Get the vocal events file for this deployment
					print('\t\tgetting vocal events...')
					vocal_events_df = pd.read_csv(os.path.join(vocal_events_root, ('_').join([deployment, 'box' + str(box), 'segments.csv'])))

					# Correct the vocalization times
					print('\t\tcorrecting times...')
					for column in ['audiomoth_start_seconds', 'audiomoth_stop_seconds', 'audiomoth_timestamp_datetime']:
						
						vocal_events_df[column+'_adjusted'] = vocal_events_df.apply(
							lambda row: correct_times(row, column ,time_correction_df=time_corrections_df), axis=1
						)

					vocal_events_df['deployment_correction_seconds'] = time_corrections_df['deployment_correction_seconds'][(time_corrections_df['deployment'] == deployment) & (time_corrections_df['box'] == box)].iloc[0]
					vocal_events_df['recovery_correction_seconds'] = time_corrections_df['recovery_correction_seconds'][(time_corrections_df['deployment'] == deployment) & (time_corrections_df['box'] == box)].iloc[0]
					vocal_events_df['estimated_or_actual_time_correction'] = time_corrections_df['atual_or_estimated'][(time_corrections_df['deployment'] == deployment) & (time_corrections_df['box'] == box)].iloc[0]
					
					updated_vocal_events.append(vocal_events_df)

					if save:
						print('\t\tsaving...')
						vocal_events_df.to_csv(os.path.join(vocal_events_root, ('_').join([deployment,'box'+str(box),'time-adjusted.csv'])), index=False)
	print('done.')
	return vocal_events_df

def correct_times(row, column, time_correction_df):
	return time_correction(row[column], row['deployment'], row['box'], time_correction_df)


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


        
        
        
    
    
    
    