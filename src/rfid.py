#this file contains functions for integrating RFID and audiomoth datasets
from datetime import date, datetime, timedelta
from scipy.stats import spearmanr
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import igraph as ig
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
import warnings
import glob
import re
import os

tqdm.pandas()

from src import parameters as prms
from src.parameters import load_json, save_json
from src.timestamps import check_if_recorded, get_season_from_date, get_audiomoth_from_box, get_boxes_from_deployment

def get_recorded_box_metadata(row, stays_df, meets_df, sex_dict):
	"""
	Use this with apply on a dataframe that has one row per box per deployment with recording start and stop as columns
	"""

	# the stays in this box in this deployment
	filtered_stays = stays_df[(stays_df['entry_time'] >= row['recording_start']) &
							  (stays_df['exit_time'] <= row['recording_stop']) &
							  (stays_df['box'] == row['box'])]

	# the mice who used the box during the deployment
	mice = list(filtered_stays['transponder_id'].unique())

	# their info
	sexes = [sex_dict.get(int(i), float('NaN')) for i in mice]
	total_mice = len(mice)
	total_males = sexes.count('M')
	total_females = sexes.count('F')
	sex_ratio = (total_males/total_mice) if total_mice != 0 else 0
		
	return pd.Series([mice, sexes,total_mice,total_males,total_females,sex_ratio])


def popcheck_id_to_database_id(popID):
	"""
	Convert ID format from pop check style to cloud of mice database style
	Translated from the R code convertid

	Arguments:
		popID (str): ID in pop check style

	Returns
		dbID (str): ID in database style


	"""

	if isinstance(popID, str):
		dbID = popID.lower()
		dbID = ('-').join([dbID[0:2],dbID[2:4],dbID[4:6], dbID[6:8], dbID[8:10]])
		return dbID
	else: 
		return float('NaN')
    
   
    
    
def database_id_to_popcheck_id(dbID):
    """
    Convert ID format from cloud of mice database style to pop check style 
    Translated from the R code convertid2
    
    Arguments:
        dbID (str): ID in database style
        
    Returns
        popID (str): ID in pop check style
         
    """

    popID = dbID.upper()
    popID = popID.replace('-','')
    
    return popID
    
def get_boxes_from_quadrant(quadrant):
    assert quadrant in [1,2,3,4], "quadrant doesn't exist"
    
    if quadrant == 1:
        return [2, 4, 6, 8, 10]
    
    elif quadrant == 2:
        return [12, 14, 16, 18, 20]
    
    elif quadrant == 3:
        return [22, 24, 26, 28, 30]
    
    elif quadrant == 4:
        return [32, 34, 36, 38, 40]
    
def get_quadrant_from_box(box):
    assert box in np.arange(2,42,2), "box number doesn't exist"
    
    if box in [2, 4, 6, 8, 10]:
        return 1
    
    elif box in [12, 14, 16, 18, 20]:
        return 2
    
    elif box in [22, 24, 26, 28, 30]:
        return 3
    
    elif box in [32, 34, 36, 38, 40]:
        return 4
    
    
def get_exits(audio_date_time, rfid_df):

    # for a given 55s recording, get how many mice existed the box
	time_to_add = 55  
    recording_start = audio_date_time
    recording_stop = recording_start + timedelta(seconds=time_to_add)
    
    #get the entrance times in this window (recording_start, recording_stop)
    rfid_df['exit_time'] = pd.to_datetime(rfid_df['exit_time'])
    exits = rfid_df.loc[((rfid_df['exit_time'] > recording_start)) & (rfid_df['exit_time'] < recording_stop)]
    
    #return
    return exits
 
def get_entrances(audio_date_time, rfid_df):
 

    # for a given 55s recording, get how many mice existed the box
	time_to_add = 55
    recording_start = audio_date_time
    recording_stop = recording_start + timedelta(seconds=time_to_add)
    
    #get the entrance times in this window (recording_start, recording_stop)
    rfid_df['entry_time'] = pd.to_datetime(rfid_df['entry_time'])
    entrances = rfid_df.loc[(rfid_df['entry_time'] > recording_start) & (rfid_df['entry_time'] < recording_stop)]
    
    #return
    return entrances

def get_occupants_in_interval(audio_date_time, stays_df):
	#get how many mice are in the box during a given 55 s recording

	# Assume recordings are 55 seconds long for now
	time_to_add = 55 

	#define start and stop of the recording
	recording_start = audio_date_time
	recording_stop = recording_start + timedelta(seconds=time_to_add)

	#look for rows where entry_time is before recording start and exit_time is after
	#get the entrance times in this window (recording_start, recording_stop)
	stays_df['entry_time'] = pd.to_datetime(stays_df['entry_time'])
	stays_df['exit_time'] = pd.to_datetime(stays_df['exit_time'])
	occupants = stays_df[(stays_df['entry_time'] <= recording_end) & (stays_df['exit_time'] >= recording_start)]

	return occupants

def get_occupants_at_time(timestamp, box, stays_df):
	"""
	Give a timestamp, a box, and stays_df; get the ids in that box at that timestamp
	"""
	
	# make sure entrance/exit times are in datetime format
	stays_df['entry_time'], stays_df['exit_time'] = pd.to_datetime(stays_df['entry_time']), pd.to_datetime(stays_df['exit_time'])
	
	# subset for this box
	stays_df = stays_df[stays_df['box'] == box]
	
	# get the occupants who entered before the timestamp and exited after the timestamp, exclusive (ie, entrance/exit shouldn't occur exactly at timestamp)
	occupants = stays_df['transponder_id'][(stays_df['entry_time'] < timestamp) & 
										   (stays_df['exit_time'] > timestamp)].to_list()
	
	return occupants
	
	
	

def determine_meeting_type(row):
    if pd.isnull(row['sex1']) or pd.isnull(row['sex2']):
        return np.nan
    elif (row['sex1'] == 'M' and row['sex2'] == 'F') or (row['sex1'] == 'F' and row['sex2'] == 'M'):
        return 'MF'
    elif row['sex1'] == 'M' and row['sex2'] == 'M':
        return 'MM'
    elif row['sex1'] == 'F' and row['sex2'] == 'F':
        return 'FF'
    else:
        return np.nan
	
def get_initial_occupants(rfid_df, box_df, box):
    #get the number of mice in the box pre to the first entrance in an rfid dataframe
    #find the box you want
    print(box_df)
    box_df['box'] = [fk_id_to_box_id(fk_id) for fk_id in box_df['box_id']]
    print(box_df)
    box_df = box_df[box_df['box']==box]
    
    #find the number of occupants after the first entrance in the rfid df (subtract 1 to get number pre to 1st entrance)
    first_entrance = rfid_df.loc[rfid_df['entry_time']==rfid_df['entry_time'].min(), 'entry_id'].values[0]
    initial_occupants = box_df.loc[box_df['event_id'] == first_entrance, 'num_partners_after_event'].values[0] - 1 
    
    return initial_occupants
    
    
def fk_id_to_box_id(fk_id):
    """
    Convert the table key for a box into it's box number
    """
    
    conversion_dict={79:2, 
                     74:4, 
                     84:6, 
                     73:8, 
                     67:10, 
                     62:12, 
                     57:14, 
                     53:16, 
                     69:18, 
                     52:20, 
                     48:22, 
                     63:24, 
                     59:26, 
                     50:28, 
                     85:30, 
                     47:32, 
                     87:34, 
                     83:36, 
                     66:38, 
                     68:40 
    }
    
    box_number = conversion_dict.get(fk_id, float('NaN'))
    return box_number

def box_id_to_fk_id(box_number):
    """
    Convert the table key for a box into it's box number
    """
    
    conversion_dict={2: 79, 
                     4: 74, 
                     6: 84, 
                     8: 73, 
                     10: 67, 
                     12: 62, 
                     14: 57, 
                     16: 53, 
                     18: 69, 
                     20: 52, 
                     22: 48, 
                     24: 63, 
                     26: 59, 
                     28: 50,
                     30: 85, 
                     32: 47, 
                     34: 87, 
                     36: 83, 
                     38: 66, 
                     40: 68}

    
    fk_id_number = conversion_dict[box_number]
    return fk_id_number
def get_test_transponder(as_database=True):
    """
    Return the test transponder ID in either database format or popcheck
    """
    if as_database:
        return "00-06-38-d0-41"
    else:
        return "000638D041"

def get_cooccupancy_matrix(mice, meets_df, stays_df):
	num_mice = len(mice)
	matrix = np.full((num_mice, num_mice), np.nan)
	cos_list = []

	for i in range(num_mice):
		for j in range(i + 1, num_mice):
			mouse1 = mice[i]
			mouse2 = mice[j]
			cos, _, _ = get_cooccupancy_index(mouse1, mouse2, meets_df, stays_df)
			matrix[i, j] = cos
			matrix[j, i] = cos  # Fill the symmetric position
			
			
	scores = matrix[np.tril_indices(matrix.shape[0], k=-1)]

	return matrix, scores

def get_igraph_network(meets_df, stays_df=None, weight=True):
    # Builds a network from the meeting dataframe
    # Always returns a weighted network - either number of contacts if weight is False
    # or duration of contacts if weight is True
    # sparse is not used
    # weight should be True
    # meets_df is the meets df
    # stays_df is the staydf
    
    meets_df=meets_df[["id1","id2","time_in_secs"]]

    if stays_df is not None:
        
        stays_df = stays_df.rename(columns={'transponder_id':'id1'}) #rename so the stays and meets have the same id names
        staytimes = stays_df.groupby('id1')['time_in_secs'].sum().reset_index()
        staytimes['starttime'] = stays_df.groupby('id1')['entry_time'].min().values
        staytimes['endtime'] = stays_df.groupby('id1')['exit_time'].max().values
        staytimes['stays'] = stays_df.groupby('id1')['time_in_secs'].max().values
        staytimes['days'] = stays_df.groupby('id1')['entry_time'].apply(lambda x: len(pd.unique(pd.to_datetime(x).dt.date)))
        fgraph = ig.Graph.DataFrame(edges=meets_df, directed=False, vertices=staytimes, use_vids=False)

    else:
        
        fgraph = ig.Graph.DataFrame(edges=meets_df, directed=False, use_vids=False)
        {v['name']: v.index for v in list(Gm.vs)}

    if weight:
        
        fgraph.simplify(multiple=True, loops=True, combine_edges={"time_in_secs": "sum"})
        
    else:
        
        fgraph.simplify(multiple=True, loops=True, combine_edges={"time_in_secs": "length"})
    
    if stays_df is not None:
        
        d1 = as_long_data_frame(fgraph)
        x = d1['time_in_secs'] # total time spent toegther for each unique pair of mice (mouse1 and mouse2)
        YA = d1['from_time_in_secs'].astype(float) - d1['time_in_secs'] #total time mouse 1 spent in any box
        YB = d1['to_time_in_secs'].astype(float) - d1['time_in_secs'] #total time mouse 2 spent in any box
        co_occupancy_score = x / (x + YA + YB) # cooccupancy score = time together/(time together + total time in boxes)
        fgraph.es['weight'] = co_occupancy_score

    return fgraph

def as_long_data_frame(graph):
    """
    Convert an igraph graph object to a long-format pandas DataFrame.

    Args:
        graph (igraph.Graph): The igraph graph object to convert.

    Returns:
        pandas.DataFrame: The long-format DataFrame.
    """
    if not isinstance(graph, ig.Graph):
        raise ValueError("Not a graph object")

    if graph.is_named():
        rn = graph.vs["name"]
    else:
        rn = np.arange(graph.vcount())

    el = np.array(graph.get_edgelist())
    edg = pd.DataFrame({"from": el[:,0], "to": el[:,1]})
    ver = pd.DataFrame()

    ver_attrs = graph.vs.attributes()
    edg_attrs = graph.es.attributes()

    for attr in ver_attrs:
        ver[f"from_{attr}"] = graph.vs[el[:,0]][attr]   
        ver[f"to_{attr}"] = graph.vs[el[:,1]][attr]

    edg['time_in_secs'] = graph.es['time_in_secs']

    long_data = pd.DataFrame()

    for col in edg.columns:
        long_data[col] = edg[col]
    for col in ver.columns:
        long_data[col] = ver[col]

    return long_data
    
def get_first_enterer_and_first_exiter(meets_df, meet_ID):
	"""
	Give a single row of a meets dataframe, get a dictionary with the IDs that entered first and exited first
	"""
	
	# get info for this meet
	this_meet = meets_df[meets_df['id'] == meet_ID]
	assert len(this_meet) == 1, "There are two meets with the same ID - do you have duplicate rows?"
	
	# find who entered first and who entered second
	if this_meet['entry1'].iloc[0] < this_meet['entry2'].iloc[0]:
		first_enterer, second_enterer = this_meet['id1'].iloc[0], this_meet['id2'].iloc[0]
	else:
		first_enterer, second_enterer = this_meet['id2'].iloc[0], this_meet['id1'].iloc[0]
		
	# find who left first and who left second
	if this_meet['exit1'].iloc[0] < this_meet['exit2'].iloc[0]:
		first_exiter, second_exiter = this_meet['id1'].iloc[0], this_meet['id2'].iloc[0]
	else:
		first_exiter, second_exiter = this_meet['id2'].iloc[0], this_meet['id1'].iloc[0]
	
	return {'first_enterer': first_enterer, 'first_exiter': first_exiter, 'second_enterer': second_enterer, 'second_exiter': second_exiter}

def count_sexes(list_of_occupant_lists, sex_dict, sex_to_count):
	"""
	Give a list of occupant ID lists (eg, occupants in each minute of a recording) a dictionary mapping ID to sex, and the sex you want to count
	(M, F, or unkown sex), get a list of numbers, where each number is the number of Male, Female, or mice with unknown sex in each inner list
	"""

	assert sex_to_count in ['M', 'F', 'unknown']

	counts = []
	if sex_to_count == 'M':

		for inner_list in list_of_occupant_lists:
			count = sum(1 for ID in inner_list if sex_dict.get(ID, float('NaN')) in ['M', 'm', 'qM', 'Mq'])
			counts.append(count)

	elif sex_to_count == 'F':

		for inner_list in list_of_occupant_lists:
			count = sum(1 for ID in inner_list if sex_dict.get(ID, float('NaN')) in ['F', 'f', 'qF', 'Fq'])
			counts.append(count)

	elif sex_to_count == 'unknown':

		for inner_list in list_of_occupant_lists:
			count = sum(1 for ID in inner_list if sex_dict.get(ID, float('NaN')) not in ['F', 'f', 'qF', 'Fq', 'M', 'm', 'qM', 'Mq'])
			counts.append(count)

	return counts


def calculate_coi_vocal_correlations(root, deployment, recording_windows, all_meets, stays, v_events, min_meets, sex_dict, save_dir, random_state, analysis_window, verbose=True):
	
	# skip if you already have a csv for this deployment
	save_path = os.path.join(save_dir, f"{deployment}.csv")
	if os.path.exists(save_path):
		print(f'Already processed deployment {deployment}')
		return

	#print(f'deployment is: {deployment}')
	
	# get the boxes recorded during this deployment
	these_recording_windows = recording_windows[deployment]
	boxes = [int(box) for box in these_recording_windows]

	# get the pairwise meetings that took place in those boxes during the time they were recorded
	recorded_meets = pd.concat(
		all_meets[(all_meets['overlap_start_time'] >= pd.to_datetime(these_recording_windows[str(box)][0])) &
			  (all_meets['overlap_start_time'] <= pd.to_datetime(these_recording_windows[str(box)][1])) &
			  (all_meets['box'] == box)]
		for box in boxes
	).reset_index()

	# get the vocalizations recorded in these boxes for this deployment
	these_v_events = v_events[(v_events['deployment'] == deployment)]
	
	# skip if there are no meets
	if recorded_meets.empty:
		print(f'Skipping deployment due to no meets... {deployment}')
		return

	# get each unique pair of mice that met in the recorded boxes and make a dictionary mapping the pair to each id
	pairs = [sorted(pair) for pair in zip(recorded_meets['id1'], recorded_meets['id2'])]
	unique_pairs = [list(pair) for pair in set(map(frozenset, pairs))]
	pair_id_dict = {('_').join(pair): ID for ID, pair in enumerate(unique_pairs)}

	# optionally print some info about the pairs that met
	if verbose:
		print(f"{len(unique_pairs)} unique pairs met in boxes {recorded_meets['box'].unique()} during the {deployment} deployment.")
		
	# do some sanity checks
	assert recorded_meets['overlap_start_time'].min() >= pd.to_datetime(deployment.split('-')[0])
	assert recorded_meets['overlap_start_time'].max() <= pd.to_datetime(deployment.split('-')[1]) + timedelta(days=1)
	assert recorded_meets[['stay1_id', 'stay2_id', 'overlap_start_time', 'overlap_end_time']].duplicated().sum() == 0
	if verbose:
		print('Data checked...\n-----------------------------------------------------------------------\n')

	# now do the analysis
	all_deployment_pairs = [] # a list to hold the pairs from this deployment
	
	# for each unique pair
	for i, pair in tqdm(enumerate(unique_pairs), desc=f"Processing unique pairs for {deployment}", total=len(unique_pairs)):
		
		if verbose:
			print('pair:', pair)
		# get the audio recoded meets between this pair if there are any, and skip the pair if not
		this_pair_recorded_meets = recorded_meets[
			(recorded_meets['id1'].isin(pair)) &
			(recorded_meets['id2'].isin(pair))
		]
		if this_pair_recorded_meets.empty:
			continue

		# get all of the meets between this pair
		this_pair_all_meets = all_meets[
			(all_meets['id1'].isin(pair)) &
			(all_meets['id2'].isin(pair))
		]
		
		#compute stats for this pair
		vocalization_data_dict = get_coi_vocalization_data_for_pair(
			pair,
			this_pair_recorded_meets,
			these_v_events,
			this_pair_all_meets,
			stays,
			analysis_window
		)

		#update the meets df for this pair with this new information on COI, vocal counts, and shuffled vocal counts during each meeting
		this_pair_recorded_meets = update_social_vocal_correlation_dataframe(
			this_pair_recorded_meets,
			vocalization_data_dict
		)
		
		#calculate the spearman correlations if there are enough recorded meetings and add this information as columns to this_pair_recorded_meets
		if len(this_pair_recorded_meets) >= min_meets:
			this_pair_recorded_meets = get_coi_vocal_correlation_for_pair(this_pair_recorded_meets, random_state, verbose)

		#collect the data for this pair and go to the next
		all_deployment_pairs.append(this_pair_recorded_meets)

	#now collect all processed "this_pair_meets" dataframes for this deployment and save it to a csv
	all_meets_with_vocs = pd.concat(all_deployment_pairs)
	
	# at least one deployment has no pairs with more than 5 meets, so you need to make the columns here and fill them with NaN
	if 'squeak_counts-next_COI_pre_correlation' not in all_meets_with_vocs.columns:
			for vocal_type in ['squeak_counts', 'USV_counts']:
				for COI_type in  ['COI_pre', 'next_COI_pre']:
					all_meets_with_vocs[f'{vocal_type}-{COI_type}_correlation'] = float('NaN')
					all_meets_with_vocs[f'{vocal_type}-{COI_type}_correlation_pvalue'] = float('NaN')
	all_meets_with_vocs.to_csv(save_path, index=False)

	if verbose:
		print(f'done with deployment: {deployment}')

def get_coi_vocalization_data_for_pair(pair, this_pair_recorded_meets, these_v_events, this_pair_all_meets, stays, analysis_window):
	"""
	Give a pair of mice, their meetings, and vocal events.
	Get a dictionary with 
	    the co-occupancy indices before/after each meeting, 
	    the actual vocal events (squeak and USV counts during each meeting)
		the shuffled vocal events (squeak and USV counts shuffled across all meetings)
	"""
	
	this_pair_meeting_info = {
		'COI_pre_list': [],
		'squeak_counts': [],
		'USV_counts': [],
		'shuffled_squeak_counts': [],
		'shuffled_USV_counts': []
	}

	if not these_v_events.empty:
		
		for _, row in this_pair_recorded_meets.iterrows(): # for each audio recorded meeting

			meet_box, meet_start, meet_stop = row['box'], row['overlap_start_time'], row['overlap_end_time']

			# Get squeak counts 
			this_pair_meeting_info['squeak_counts'].append(
				len(these_v_events[
					(these_v_events['audiomoth_start_seconds_adjusted'] > meet_start) & 
					(these_v_events['audiomoth_start_seconds_adjusted'] < meet_stop) & 
					(these_v_events['label'] == 'squeak') & 
					(these_v_events['box'] == meet_box)
				])
			)

			# Get USV counts
			this_pair_meeting_info['USV_counts'].append(
				len(these_v_events[
					(these_v_events['audiomoth_start_seconds_adjusted'] > meet_start) & 
					(these_v_events['audiomoth_start_seconds_adjusted'] < meet_stop) & 
					(these_v_events['label'] == 'USV') & 
					(these_v_events['box'] == meet_box)
				])
			)

			# this performs the core co-occupancy index (COI) calculations
			this_pair_meeting_info['COI_pre_list'].append(get_COI_in_window(stays_df=stays, 
															meets_df=this_pair_all_meets, # note meets here is ALL meets, not just those recorded
															pair=pair,
															window=analysis_window,
															meet_ID=row['id']))


		#these lines resample squeak and USV counts randomly, ie "shuffling" them wrt to each meet, then add them back to their respective shuffled columns
		this_pair_meeting_info['shuffled_squeak_counts'] = pd.Series(this_pair_meeting_info['squeak_counts']).sample(frac=1, random_state=123456).values
		this_pair_meeting_info['shuffled_USV_counts'] = pd.Series(this_pair_meeting_info['USV_counts']).sample(frac=1, random_state=123456).values

	return this_pair_meeting_info

def get_COI_in_window(stays_df, meets_df, pair, window, meet_ID):
	"""
	Give an unfiltered stays df, an unfiltered meets df, a pair of mice, a meeting ID (meets_window -- this is the id of the meeting row in the meets table), the duration of a window in hours over which you want to calculate COI before the start of a meet (COI_window)

	Get a COI for this pair in this window.

	This function is so fucked up because it needs to deal with edge cases where the window edge falls within a stay or meet corresponding to the given meet.
	It does this by clipping the corresponding stay and meet to ensure there are always stays from each mouse for all meets in the meet window.

	"""
	
	# check the ID for the meeting
	assert meet_ID is not None, " A meet_ID must be provided."
	assert meet_ID in meets_df['id'].unique(), "meet ID is not in meets_df"

	# get info for this meet and check that the meet ID is unique (if not you may have duplicated rows)
	this_meet = meets_df[meets_df['id'] == meet_ID]
	assert len(this_meet) == 1, "There are two meets with the same ID - do you have duplicate rows?"
	meet_pair = [this_meet['id1'].iloc[0], this_meet['id2'].iloc[0]]
	assert sorted(pair) == sorted(meet_pair) # I'm just curious if any of these don't agree

	#unpack
	meet_start, meet_stop = this_meet['overlap_start_time'].iloc[0], this_meet['overlap_end_time'].iloc[0]

	#subset meets and stays for these mice
	meets_df = meets_df[meets_df['id1'].isin(pair) & meets_df['id2'].isin(pair)]
	stays_df = stays_df[stays_df['transponder_id'].isin(pair) | stays_df['transponder_id'].isin(pair)]

	#sort
	meets_df = meets_df.sort_values(by = ['overlap_start_time'])
	stays_df = stays_df.sort_values(by = ['entry_time'])
		
	# the window starts "window" hours prior to the meet start
	window_start = meet_start - timedelta(hours=window)

	## GET THE MEETS prior to this meet

	#check if the edge of the window falls within a meeting
	#only care about exclusive inequality bc if meet edge is by chance identical to window edge, 
	#it has effectively already been "clipped"
	window_in_meet = meets_df[(meets_df['overlap_start_time'] < window_start) 
							  & (meets_df['overlap_end_time'] > window_start) 
							 ]

	#if not, things are easier
	if len(window_in_meet) == 0:

		# just get the meetings prior to this meet that fall within the analysis window
		# inclusive inequality for consistency, but doesn't need to be
		meets_pre = meets_df[(meets_df['overlap_end_time'] <= meet_start) 
							   & (meets_df['overlap_start_time'] >= window_start)
							  ]

	#if so, things are harder - clip the meet so that only the portion within window contributes to COI
	elif len(window_in_meet) != 0:

		# there can only be one
		assert len(window_in_meet) == 1, "There can only be one meet between these mice that spans the window edge"

		# find the meet ID and make sure it is the in the big meetings df
		spanning_meet = window_in_meet['id'].iloc[0]
		assert spanning_meet in meets_df['id'].unique(), "spanning meet ID is not in meets_df"

		# change the start of this meet to the start of the window (clip it)
		meets_df.loc[meets_df['id'] == spanning_meet, 'overlap_start_time'] = window_start

		# now get the meets pre like above
		# inclusive bc you have just clipped a meet edge to be identical to a window edge and want to keep this meeting
		meets_pre = meets_df[(meets_df['overlap_end_time'] <= meet_start) 
							   & (meets_df['overlap_start_time'] >= window_start)
							  ]


	## GET THE STAYS prior to this meet
	## Slightly more complicated than getting the meets, bc (for pre calculation) you need to exclude portions of stays
	## that extend into the meeting for which you are calculatiing COI. You do this will the function
	## clip_stays_to_meets

	#check if the edge of the window falls within a stay by either mouse
	#use exclusive inequality in bc if stay edge is by chance identical to window edge, it has already been "clipped"
	window_in_stay = stays_df[(stays_df['entry_time'] < window_start) 
							  & (stays_df['exit_time'] > window_start) 
							 ]

	# if not, things are easier
	if len(window_in_stay) == 0:

		# get the stays pre

		#first clip the stay of the mouse who entered first, so your COI calculation does not include the portion of the stay 
		#occurring during the meet
		stays_pre = clip_stay_to_meet(stays_df, meets_df, meet_ID = meet_ID)

		#now subset 
		#inclusive bc you have just clipped a stay edge to be identical to the meet edge and want to keep this clipped stay
		stays_pre = stays_pre[(stays_pre['entry_time'] >= window_start) 
							   & (stays_pre['exit_time'] <= meet_start)
							  ]

	#if so, things are harder - clip the stays so that only the portion within window contributes to COI
	elif len(window_in_stay) != 0:

		# there can be at most 2, one for each mouse in the pair (stays can be in separate boxes)
		assert len(window_in_stay) <= 2, "There can be at most 2 stays spanning a window edge"

		# clip whichever stay needs to be clipped (or both)
		for mouse in window_in_stay['transponder_id'].unique():

			# find the stay and make sure it is in the big stays df
			# if the mouse does not have a spanning stay, it will not be in the window_in_stay df
			spanning_stay = window_in_stay['id'][window_in_stay['transponder_id'] == mouse].iloc[0] 
			assert spanning_stay in stays_df['id'].unique(), "spanning stay ID is not in stays_df"

			# change the start of this stay to the start of the window (clip it)
			stays_df.loc[stays_df['id'] == spanning_stay, 'entry_time'] = window_start

		# get the stays pre, clipping the stay of the mouse who entered first, so your COI calculation does not include this stay
		stays_pre = clip_stay_to_meet(stays_df, meets_df, meet_ID = meet_ID)

		#now subset 
		#inclusive bc you have just clipped a stay edge to be identical to the window/meet edge and want to keep this clipped stay
		stays_pre = stays_pre[(stays_pre['entry_time'] >= window_start) 
							   & (stays_pre['exit_time'] <= meet_start)
							  ]

	## GET THE COI (co-occupancy index) 
	## This is the ratio (time mouse A and mouse B together) / (time together + time A in a box apart from B + time B in a box apart from A)

	# the worst is over - if the mice met after this meeting, just get the COI and return it
	if len(meets_pre) > 0:
		COI_pre, YA, YB = get_cooccupancy_index(pair[0], pair[1], meets_pre, stays_pre)
		assert YA >= 0, "There may be a stay missing - meeting time longer than total stay time for " + pair[0] 
		assert YB >= 0, "There may be a stay missing - meeting time longer than total stay time for " + pair[1] 
	else:
		COI_pre = 0
	return COI_pre


def get_cooccupancy_index(mouse1, mouse2, meets_df, stays_df):
	"""
	Get the coocupancy score used by Julian Evans in previous Lindholm/König lab pubs 
	This value is the edge weight in graphs generated by get_igraph_network

	"""
	
	# Use sets for faster membership check
	mice_set = {mouse1, mouse2}

	# Group stays by mouse id using a dictionary
	stays_dict = stays_df.groupby('transponder_id')['time_in_secs'].sum().to_dict() # supposed to be a fast way to do this

	#get total time these mice spent together
	time_together = float(meets_df.query("id1 in @mice_set and id2 in @mice_set")['time_in_secs'].sum()) # supposed to be a fast way to do this

	# Calculate total time mouse 1 spent in any box NOT with mouse 2
	assert mouse1 in stays_dict.keys(), mouse1+" is not in the stays_df. Is your meets_df empty?"
	YA = round(float(stays_dict[mouse1]) - time_together, 5) # 0 if they are the same to 5 decimals -- needed to deal with use of float

	# Calculate total time mouse 2 spent in any box NOT with mouse 1 
	assert mouse2 in stays_dict.keys(), mouse2+" is not in the stays_df. Is your meets_df empty?"
	YB = round(float(stays_dict[mouse2]) - time_together, 5) # 0 if they are the same to 5 decimals -- needed to deal with use of float

	# Calculate cooccupancy score = time together / (time together + time apart)
	# This score is 0 if mice never meet (time_together = 0) and 1 if they spend all their time together (time apart = 0)
	# It approaches 0 as mice spend more time apart relative to time together (time apart >> time together)
	total_time = time_together + YA + YB
	co_occupancy_score = time_together / total_time if total_time != 0 else 0

	return co_occupancy_score, YA, YB #so you can check if YA or YB is negative

def clip_stay_to_meet(stays_df, meets_df, meet_ID):
	"""
	Clip a stay corresponding to a given meet_ID - internal helper function
	"""

	# get info for this meet
	this_meet = meets_df[meets_df['id'] == meet_ID]
	assert len(this_meet) == 1, "There are two meets with the same ID - do you have duplicate rows?"
	
	# unpack
	meet_start, meet_stop = this_meet['overlap_start_time'].iloc[0], this_meet['overlap_end_time'].iloc[0]

	# now deal with the stay of the mouse that entered first in this meet - you need to clip it so it ends before the start of the meet
	# otherwise this stay will contribute this mouses stays prior to this meet, which is not true. The portion of the stay you are dropping 
	# occurredd DURING the meet. That's why it needs to be ignored
	
	# get the stay_ID corresponding to the first entrance of the meet
	if this_meet['entry1'].iloc[0] < this_meet['entry2'].iloc[0]:
		first_entrance_stay_ID, second_entrance_stay_ID = this_meet['stay1_id'].iloc[0], this_meet['stay2_id'].iloc[0]
	else:
		first_entrance_stay_ID, second_entrance_stay_ID = this_meet['stay2_id'].iloc[0], this_meet['stay1_id'].iloc[0]
		
	# get the stay ID corresponding to the second exit of the meet
	if this_meet['exit1'].iloc[0] < this_meet['exit2'].iloc[0]:
		first_exit_stay_ID, second_exit_stay_ID = this_meet['stay1_id'].iloc[0], this_meet['stay2_id'].iloc[0]
	else:
		first_exit_stay_ID, second_exit_stay_ID = this_meet['stay2_id'].iloc[0], this_meet['stay1_id'].iloc[0]

	# clip the stay of the first enterer so it ends before the meet starts
	stays_df.loc[stays_df['id'] == first_entrance_stay_ID, 'exit_time'] = meet_start
	return stays_df

	
def update_social_vocal_correlation_dataframe(this_pair_recorded_meets, vocalization_data_dict):
    this_pair_recorded_meets['shuffled_squeak_counts'] = vocalization_data_dict['shuffled_squeak_counts']
    this_pair_recorded_meets['shuffled_USV_counts'] = vocalization_data_dict['shuffled_USV_counts']
    this_pair_recorded_meets['squeak_counts'] = vocalization_data_dict['squeak_counts']
    this_pair_recorded_meets['USV_counts'] = vocalization_data_dict['USV_counts']
    this_pair_recorded_meets['COI_pre'] = vocalization_data_dict['COI_pre_list']
    this_pair_recorded_meets['next_COI_pre'] = this_pair_recorded_meets['COI_pre'].shift(-1)
    this_pair_recorded_meets['num_meets'] = len(this_pair_recorded_meets)
    return this_pair_recorded_meets

def get_coi_vocal_correlation_for_pair(this_pair_recorded_meets, random_state, verbose):

	# use all available cores
	num_cores = cpu_count()

	this_pair_recorded_meets = this_pair_recorded_meets.sort_values(by=['overlap_start_time'])
	
	data = this_pair_recorded_meets[['shuffled_squeak_counts', 'shuffled_USV_counts', 'squeak_counts', 'USV_counts', 'COI_pre', 'next_COI_pre']].dropna()

	vocal_data = this_pair_recorded_meets[['shuffled_squeak_counts', 'shuffled_USV_counts', 'squeak_counts', 'USV_counts', ]]
	
	COI_data = this_pair_recorded_meets[['COI_pre', 'next_COI_pre']]

	# Prepare the arguments for parallel processing
	args = [
		(vocal_data, COI_data, vocal_type, COI_type)
		for vocal_type in vocal_data.columns
		for COI_type in COI_data.columns
	]

	# Use multiprocessing Pool to parallelize the computation
	with Pool(num_cores) as pool:
		results = pool.map(compute_correlation, args)

	correlation_results = {}
	for vocal_type, COI_type, correlation, p_value in results:
		correlation_results[(vocal_type, COI_type)] = (correlation, p_value)

		if verbose:
			print(f"Spearman correlation between {vocal_type} and {COI_type}: {correlation} (p-value: {p_value})")

		this_pair_recorded_meets[f'{vocal_type}-{COI_type}_correlation'] = correlation
		this_pair_recorded_meets[f'{vocal_type}-{COI_type}_correlation_pvalue'] = p_value

	return this_pair_recorded_meets

def is_constant(series):
    return series.nunique() == 1
def compute_correlation(args):
    vocal_data, COI_data, vocal_type, COI_type = args
    
    # Check if either column is constant across all rows
    if is_constant(vocal_data[vocal_type]) or is_constant(COI_data[COI_type]):
        return (vocal_type, COI_type, float('NaN'), float('NaN'))
    
    # Perform Spearman correlation
    correlation, p_value = spearmanr(vocal_data[vocal_type], COI_data[COI_type], nan_policy = 'omit')
    return (vocal_type, COI_type, correlation, p_value)

def count_perievent_vocalizations(root, v_events, m_events, window_s, event_type = 'entrance', random_seed = 123456, randomize = False):

	recording_durations = load_json(os.path.join(root, 'parameters', 'json_files','recording_durations.json'))
	
	# for controls, get the number of vocalizations in each box for each deployment, then
	# randomly sample (with replacement) that number of timestamps from the window that begins
	# at the start of the first vocalization in that box and ends at the start of the last
	if randomize:
		v_events_aligned_randomized_list = []
		np.random.seed(random_seed)

		for deployment in tqdm(v_events['deployment'].unique()):

			this_deployment = v_events[v_events['deployment'] == deployment]

			for box in this_deployment['box'].unique():
				
				moth = get_audiomoth_from_box(root, box, deployment)
				start, end = recording_durations[deployment][moth][0], recording_durations[deployment][moth][1] #start and end of the audio recording
				start, end = pd.to_datetime(start), pd.to_datetime(end) 
				recording_duration_hours = (end - start).total_seconds()/(60*60)
				if recording_duration_hours < 5: #ignore any aborted recordings just in case
					continue
				
				this_box = this_deployment[this_deployment['box'] == box]
				this_box['audiomoth_start_seconds_adjusted'] = pd.to_datetime(this_box['audiomoth_start_seconds_adjusted'])
				num_vocs = len(this_box)
				random_voc_times = np.random.uniform(start.timestamp(), end.timestamp(), num_vocs) # **this is the key line that randomizes**
				random_voc_times = pd.to_datetime(random_voc_times, unit='s')  # Convert to datetime
				this_box['audiomoth_start_seconds_adjusted'] = random_voc_times
				v_events_aligned_randomized_list.append(this_box)
				
		v_events = pd.concat(v_events_aligned_randomized_list)  

	assert 'audiomoth_timestamp_datetime_adjusted' in v_events.columns, "It looks like you are using a v_events table that hasn't been corrected using the audiomoth chime"

	assert event_type in ['entrance', 'exit'], "event_type must be one of ['entrance', 'exit']"

	perievent_vocs = []
	vocs_present = []

	for deployment in tqdm(v_events['deployment'].unique()):

		these_v_events = v_events[v_events['deployment'] == deployment]
		boxes = these_v_events['box'].unique()
		
#		start = these_v_events['audiomoth_timestamp_datetime_adjusted'].min()
#		end = these_v_events['audiomoth_timestamp_datetime_adjusted'].max()
		
		for box in boxes:
			
			moth = get_audiomoth_from_box(root, box, deployment)
			start, end = recording_durations[deployment][moth][0], recording_durations[deployment][moth][1] #start and end of the audio recording
			start, end = pd.to_datetime(start), pd.to_datetime(end) 
			recording_duration_hours = (end - start).total_seconds()/(60*60)
			if recording_duration_hours < 5:
				continue
			
			these_m_events = m_events[(m_events['event_time'] >= start) 
									  & (m_events['event_time'] <= end) 
									  & (m_events['box'] == box)]

			if event_type == 'entrance':
				events = these_m_events[these_m_events['event_type'] == 'entrance']
			else:
				events = these_m_events[these_m_events['event_type'] == 'exit']

			for mouse in these_m_events['id_triggering_event'].unique():

				these_events = events[(events['id_triggering_event'] == mouse)].copy()

				if len(these_events) > 0:

					these_events['event_time'] = pd.to_datetime(these_events['event_time'])
					for event_time in these_events['event_time']:
						
						event_time = pd.to_datetime(event_time)
						start_time = event_time - timedelta(seconds=window_s)
						end_time = event_time + timedelta(seconds=window_s)
						window_length = int((end_time - start_time).total_seconds())
						assert window_length == window_s*2, "Window length is not what it should be"
						
						vocs_in_window = these_v_events[(these_v_events['audiomoth_start_seconds_adjusted'] >= start_time) & 
														 (these_v_events['audiomoth_start_seconds_adjusted'] < end_time) & 
														 (these_v_events['box'] == box)] 

#						
#						assert vocs_in_window['audiomoth_start_seconds_adjusted'].ge(start_time).all(), "Some values are less than start_time"
#						assert vocs_in_window['audiomoth_start_seconds_adjusted'].lt(end_time).all(), "Some values are greater than or equal to end_time"
#						assert vocs_in_window['box'] == box
#						assert vocs_in_window['deployment'] == deployment
						
						# Calculate number of USVs and squeaks in interval before and after entrance within window_s
						pre = (vocs_in_window['audiomoth_start_seconds_adjusted'] >= start_time) & (vocs_in_window['audiomoth_start_seconds_adjusted'] < event_time)
						post = (vocs_in_window['audiomoth_start_seconds_adjusted'] >= event_time) & (vocs_in_window['audiomoth_start_seconds_adjusted'] < end_time)
						
						num_USVs_prior = len(vocs_in_window[ pre & (vocs_in_window['label'] == 'USV')])
						num_squeaks_prior = len(vocs_in_window[pre & (vocs_in_window['label'] == 'squeak')])
						num_USVs_post = len(vocs_in_window[ post & (vocs_in_window['label'] == 'USV')])
						num_squeaks_post = len(vocs_in_window[post & (vocs_in_window['label'] == 'squeak')])

						# Collect data
						vocs_present.append({
							'deployment': deployment,
							'event_time': event_time,
							'box': box,
							'mouse': mouse,
							'num_USVs_in_interval_prior': num_USVs_prior,
							'num_squeaks_in_interval_prior': num_squeaks_prior,
							'num_USVs_in_interval_post': num_USVs_post,
							'num_squeaks_in_interval_post': num_squeaks_post,
							'event_type': event_type,
							'randomized': randomize
						}) 

						#if there are any, get latencies to actual and randomized vocaliations
						if not vocs_in_window.empty:

							np.random.seed(random_seed)  # Set the seed
							num_vocs = len(vocs_in_window)
							random_voc_times = np.random.uniform(start_time.timestamp(), end_time.timestamp(), num_vocs)
							random_voc_times = pd.to_datetime(random_voc_times, unit='s')  # Convert to datetime
							vocs_in_window['audiomoth_start_seconds_adjusted_randomized'] = random_voc_times

							# add latencies
							vocs_in_window['latency_to_voc'] = (vocs_in_window['audiomoth_start_seconds_adjusted'] - event_time).apply(lambda x: x.total_seconds())
							
							actual_vocs_following = vocs_in_window['latency_to_voc'][vocs_in_window['latency_to_voc'] > 0]
							actual_vocs_preceding = vocs_in_window['latency_to_voc'][vocs_in_window['latency_to_voc'] < 0]
							vocs_in_window['latency_to_first_voc_following'] = actual_vocs_following.min() if not actual_vocs_following.empty else float('NaN')
							vocs_in_window['latency_to_first_voc_preceding'] = actual_vocs_preceding.max() if not actual_vocs_preceding.empty else float('NaN')

							
							vocs_in_window['latency_to_randomized_voc'] = (vocs_in_window['audiomoth_start_seconds_adjusted_randomized'] - event_time).apply(lambda x: x.total_seconds())
							randomized_vocs_preceding = vocs_in_window['latency_to_randomized_voc'][vocs_in_window['latency_to_randomized_voc'] < 0]
							randomized_vocs_following = vocs_in_window['latency_to_randomized_voc'][vocs_in_window['latency_to_randomized_voc'] > 0]
							vocs_in_window['latency_to_randomized_first_voc_following'] = randomized_vocs_following.min() if not randomized_vocs_following.empty else float('NaN')
							vocs_in_window['latency_to_randomized_first_voc_preceding'] = randomized_vocs_preceding.max() if not randomized_vocs_preceding.empty else float('NaN')

							# add mouse data
							vocs_in_window['mouse_triggering_event'] = mouse
							vocs_in_window['event_time'] = event_time
							vocs_in_window['start_time'] = start_time
							vocs_in_window['end_time'] = end_time
							vocs_in_window['window'] = window_s
							vocs_in_window['num_vocs_in_window'] = len(vocs_in_window)
							vocs_in_window['event_type'] = event_type
							vocs_in_window['randomized'] = randomize
							vocs_in_window['event_ID'] = str(mouse)+'_'+str(event_time)
							

							# collect the data
							perievent_vocs.append(vocs_in_window)

	return pd.concat(perievent_vocs), pd.DataFrame(vocs_present)

def bin_vocal_latencies(latency_df, bin_width, groupby):
    
    # Create an empty dictionary to store the results
    result = {}

    # Loop through each unique event_ID
    for event_id, group in tqdm(latency_df.groupby(groupby)):

        # Get the window value in seconds for this group
        window = group['window'].iloc[0]  # assuming window is constant within the same event_ID

        # Check bin is less than window
        assert bin_width < window, "Bin width must be less than window within which latencies were calculated"

        # Create the bins
        bins = np.arange(-window, window + bin_width, bin_width)

        # Define labels for the bins (midpoints)
        bin_labels_values = (bins[:-1] + bins[1:]) / 2

        # Cut latencies into bins, using the bin midpoints as labels
        bin_labels = pd.cut(group['latency_to_voc'], bins, right=False, labels=bin_labels_values)

        # Count the occurrences in each bin
        bin_counts = bin_labels.value_counts().sort_index()

        # Convert to dictionary to aggregate results
        bin_counts_dict = bin_counts.to_dict()

        # Store the counts in the result dictionary
        result[event_id] = bin_counts_dict

    # Convert result to DataFrame for better display
    result_df = pd.DataFrame(result).fillna(0).astype(int)

    return result_df

def count_perivocal_events(root, v_events, m_events, window_s, random_seed = 123456, randomize = False):
	
	v_events = v_events[v_events['label'] != 'noise']
	
	recording_durations = load_json(os.path.join(root, 'parameters', 'recording_durations.json'))
	
	# for controls, get the number of vocalizations in each box for each deployment, then
	# randomly sample (with replacement) that number of timestamps from the window that begins
	# at the start of the first vocalization in that box and ends at the start of the last
	if randomize:
		v_events_aligned_randomized_list = []
		
		
		for deployment in tqdm(v_events['deployment'].unique()):

			this_deployment = v_events[v_events['deployment'] == deployment]
			
			for box in this_deployment['box'].unique():
				
				moth = get_audiomoth_from_box(root, box, deployment)
				start_time, end_time = recording_durations[deployment][moth][0], recording_durations[deployment][moth][1] #start and end of the audio recording
				start_time, end_time = pd.to_datetime(start_time), pd.to_datetime(end_time)
				this_box = this_deployment[this_deployment['box'] == box]
				this_box['audiomoth_start_seconds_adjusted'] = pd.to_datetime(this_box['audiomoth_start_seconds_adjusted'])
				num_vocs = len(this_box)
				random_voc_times = np.random.uniform(start_time.timestamp(), end_time.timestamp(), num_vocs)
				random_voc_times = pd.to_datetime(random_voc_times, unit='s')  # Convert to datetime
				this_box['audiomoth_start_seconds_adjusted'] = random_voc_times
				v_events_aligned_randomized_list.append(this_box)
				
		v_events = pd.concat(v_events_aligned_randomized_list)  

	assert 'audiomoth_timestamp_datetime_adjusted' in v_events.columns, "It looks like you are using a v_events table that hasn't been corrected using the audiomoth chime"

	perivoc_events = []
	events_present = []

	for deployment in tqdm(v_events['deployment'].unique()):

		these_v_events = v_events[v_events['deployment'] == deployment]
		boxes = these_v_events['box'].unique()
		
#		start = these_v_events['audiomoth_timestamp_datetime_adjusted'].min()
#		end = these_v_events['audiomoth_timestamp_datetime_adjusted'].max()
		
		for box in boxes:
			
			moth = get_audiomoth_from_box(root, box, deployment)
			start, end = recording_durations[deployment][moth][0], recording_durations[deployment][moth][1] #start and end of the audio recording
			
			these_m_events = m_events[(m_events['event_time'] >= start) 
									  & (m_events['event_time'] <= end) 
									  & (m_events['box'] == box)]
			these_v_events = v_events[v_events['box'] == box]
			

			for vocalization_start, vocalization_label in zip(these_v_events['audiomoth_start_seconds_adjusted'], these_v_events['label']):

				vocalization_start = pd.to_datetime(vocalization_start)
				start_time = vocalization_start - timedelta(seconds=window_s)
				end_time = vocalization_start + timedelta(seconds=window_s)
				
				events_in_window = these_m_events[(these_m_events['event_time'] >= start_time) & 
												  (these_m_events['event_time'] < end_time) & 
												  (these_m_events['box'] == box)] 
				
				pre = (events_in_window['event_time'] >= start_time) & (events_in_window['event_time'] < vocalization_start)
						
				post = (events_in_window['event_time'] >= vocalization_start) & (events_in_window['event_time'] < end_time)
				
				num_entrances_prior = len(events_in_window[ pre & (events_in_window['event_type'] == 'entrance')])
				num_exits_prior = len(events_in_window[pre & (events_in_window['event_type'] == 'exit')])
				num_entrances_post = len(events_in_window[ post & (events_in_window['event_type'] == 'entrance')])
				num_exits_post = len(events_in_window[post & (events_in_window['event_type'] == 'exit')])
				
				# Collect data
				events_present.append({
					'deployment': deployment,
					'vocalization_time': vocalization_start,
					'box': box,
					'vocalization_label': vocalization_label,
					'num_USVs_in_interval_prior': num_entrances_prior,
					'num_squeaks_in_interval_prior': num_exits_prior,
					'num_USVs_in_interval_post': num_entrances_post,
					'num_squeaks_in_interval_post': num_exits_post,
					'randomized': randomize
				}) 
				
				
				#if there are any, get latencies to actual and randomized vocaliations
				if not events_in_window.empty:

					# add latencies
					events_in_window['latency_to_event'] = (events_in_window['event_time'] - vocalization_start).apply(lambda x: x.total_seconds())

					events_following = events_in_window['latency_to_event'][events_in_window['latency_to_event'] > 0]
					events_preceding = events_in_window['latency_to_event'][events_in_window['latency_to_event'] < 0]
					events_in_window['latency_to_first_event_following'] = events_following.min() if not events_following.empty else float('NaN')
					events_in_window['latency_to_first_event_preceding'] = events_preceding.max() if not events_preceding.empty else float('NaN')


					# add mouse data
	
					events_in_window['vocalization_time'] = vocalization_start
					events_in_window['start_time'] = start_time
					events_in_window['end_time'] = end_time
					events_in_window['window'] = window_s
					events_in_window['num_events_in_window'] = len(events_in_window)
					events_in_window['vocalization_label'] = vocalization_label
					events_in_window['randomized'] = randomize
					events_in_window['vocalization_ID'] = vocalization_label+'_'+str(vocalization_start)


					# collect the data
					perivoc_events.append(events_in_window)

	return pd.concat(perivoc_events), pd.DataFrame(events_present)
             
	
def bin_event_latencies(latency_df, bin_width, groupby):
    
    # Create an empty dictionary to store the results
    result = {}

    # Loop through each unique event_ID
    for vocal_id, group in tqdm(latency_df.groupby(groupby)):

        # Get the window value in seconds for this group
        window = group['window'].iloc[0]  # assuming window is constant within the same event_ID

        # Check bin is less than window
        assert bin_width < window, "Bin width must be less than window within which latencies were calculated"

        # Create the bins
        bins = np.arange(-window, window + bin_width, bin_width)

        # Define labels for the bins (midpoints)
        bin_labels_values = (bins[:-1] + bins[1:]) / 2

        # Cut latencies into bins, using the bin midpoints as labels
        bin_labels = pd.cut(group['latency_to_event'], bins, right=False, labels=bin_labels_values)

        # Count the occurrences in each bin
        bin_counts = bin_labels.value_counts().sort_index()

        # Convert to dictionary to aggregate results
        bin_counts_dict = bin_counts.to_dict()

        # Store the counts in the result dictionary
        result[event_id] = bin_counts_dict

    # Convert result to DataFrame for better display
    result_df = pd.DataFrame(result).fillna(0).astype(int)

    return result_df
def get_territory_size(box_list, coordinates_df):
    """
    Calculate the distance or area for the given list of box numbers.
    
    Parameters:
    - box_list: list of int, box numbers.
    - coordinates_df: pd.DataFrame, containing columns 'box', 'x', and 'y'.
    
    Returns:
    - float, distance between two boxes or area of the smallest polygon bounding the boxes or NaN.
    """
    if len(box_list) == 1:
        return np.nan
    
    # Get the coordinates of the specified boxes
    subset_df = coordinates_df[coordinates_df['box'].isin(box_list)]
    
    if len(box_list) == 2:
        # Calculate the Euclidean distance between the two boxes
        (x1, y1), (x2, y2) = subset_df[['x', 'y']].values
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance
    
    elif len(box_list) > 2:
        # Calculate the area of the smallest polygon (convex hull) bounding the boxes
        points = subset_df[['x', 'y']].values
        hull = ConvexHull(points)
        area = hull.volume  # ConvexHull.volume gives the area for 2D cases
        return area
    
    return np.nan
                        
