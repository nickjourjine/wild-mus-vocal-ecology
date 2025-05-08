#this file contains functions for dealing with time stamps from audiomoths and the barn rfid system

import os
import re
import glob
import json
import pytz
import ephem
import numpy as np
import pandas as pd
from astral import LocationInfo
from astral.sun import sun
from datetime import date, datetime, timedelta

#custom 
from src.filespaths import get_paths_raw, sort_nicely
from src.parameters import save_json, load_json
def get_missing_predictions(root, missing_dict):
    
	all_deployments = set(get_deployments(root))
	missing_from_all = all_deployments.copy()

	# Iterate over the missing deployments for each moth
	for moth, missing_deployments in missing_dict.items():
		# Convert the list to a set and update the missing_from_all set
		missing_from_all &= set(missing_deployments)

	return list(missing_from_all)

def get_playbacks_dates(root):
	"""
	Give the root directory for the project
	Get a nested dictionary of playback dates and boxes where the keys are [sound_type][date][box]
	"""
	
	dates = load_json(os.path.join(root, 'parameters', 'playback_dates.json'))
	
	return dates
	
	
def is_recovery_date(date, deployment):
    """
    check if a given date from a deployment is the date the audiomoths were recovered
    """
    date = date.date()
    dropoff_date = deployment.split('-')[0]
    dropoff_date = ('-').join([dropoff_date[:4], dropoff_date[4:6], dropoff_date[6:]])
    dropoff_date = datetime.strptime(dropoff_date, '%Y-%m-%d').date()
    if date == dropoff_date:
        return 0
    else:
        return 1
    
def make_time_columns(df, audiomoth_timestamp_column, start_seconds_column='start_seconds', stop_seconds_column='stop_seconds'):
    """
    take a df that has a column for audiomoth minute and a timestamp for an event within each minute and add
    a column for the universal time in datetime format
    """
    
    assert audiomoth_timestamp_column in df.columns
    assert start_seconds_column in df.columns
    assert stop_seconds_column in df.columns
    

    #make datetime-like string if time is audiomoth  
    audiomoth_datetime_column = ('_').join([audiomoth_timestamp_column,'datetime'])
    df[audiomoth_datetime_column] = [audiomoth_to_datetime(i) for i in df[audiomoth_timestamp_column]]

    #convert start and stop seconds to date_time
    start_times = []
    stop_times = []
    for i, start_secs, stop_secs in zip(df[audiomoth_datetime_column], df[start_seconds_column], df[stop_seconds_column]):
        start_times.append(i+timedelta(seconds=start_secs))
        stop_times.append(i+timedelta(seconds=stop_secs))
    df['_'.join(['audiomoth', start_seconds_column])]  = start_times
    df['_'.join(['audiomoth', stop_seconds_column])] = stop_times
    
    return df
    
    
def get_sunrises(location, start_date, end_date):
    """
    Give a location in [lat, lon] format, a start date and an end date,
    return a dictionary of sunrise times with dates as keys and times as values.
    """
    city = LocationInfo(name='City', region='Region', timezone='Europe/Zurich', latitude=location[0], longitude=location[1])

    sunrise_times = {}

    current_date = start_date
    while current_date <= end_date:
        s = sun(city.observer, date=current_date, tzinfo=pytz.timezone('Europe/Zurich'))
        sunrise_time = s['sunrise'].strftime('%H:%M:%S')
        sunrise_times[current_date.date()] = pd.to_datetime(sunrise_time).time()
        current_date += timedelta(days=1)

    return sunrise_times
	
def get_sunsets(location, start_date, end_date):
	"give a location in [lat,lon] format, a start date and an end date, return a list of sunset times, one for each day between start and end"

	city = LocationInfo(name='City', region='Region', timezone='Europe/Zurich', latitude=location[0], longitude=location[1])

	sunset_times = {}

	current_date = start_date
	while current_date <= end_date:
		s = sun(city.observer, date=current_date, tzinfo=pytz.timezone('Europe/Zurich'))
		sunset_time = s['sunset'].strftime('%H:%M:%S')
		sunset_times[current_date.date()] = pd.to_datetime(sunset_time).time()
		current_date += timedelta(days=1)

	return sunset_times
	
def check_sunup(time):
    """
    Take a datetime value and return whether the sun was up (1) or down (0) in Illnau at that time
    """
    
    from suntime import Sun, SunTimeException
    import pytz
    from datetime import datetime, timezone

    #get location and time zone data for Illnau, Switzerland
    latitude = 47.4273
    longitude = 8.6903
    tz_zurich = pytz.timezone('Europe/Zurich')
    sun = Sun(latitude, longitude)
    
    #convert if needed
    if not isinstance(time, datetime):
        time = datetime.fromisoformat(time)
    
    #get sundown and sunup time for the date
    date = time.date()
    
    #get sunrise and sunset time on that date in Illnau
    sunrise = sun.get_sunrise_time(date).astimezone(tz_zurich)
    sunset = sun.get_sunset_time(date).astimezone(tz_zurich)
    
    #return whether the time is during sun up or not
    
    if sunrise < time.replace(tzinfo=tz_zurich) < sunset:
        return 1
    else:
        return 0
    
def datetime_to_audiomoth(timestamp):
    #convert a timestamp format to audiomoth format
    
    #convert to string
    if isinstance(timestamp, datetime):
        timestamp = str(timestamp)
        
    assert isinstance(timestamp, str)

    date = timestamp.split(" ")[0].replace('-', '')
    time = timestamp.split(" ")[1].replace(':', '')
    audiomoth_time = ('_').join([date,time])

    return audiomoth_time
def audiomoth_to_datetime(time):
    """
    convert audiomoth time format (yyyymmdd_hhmmss) to transponder system format (yyyy-mm-dd hh:mm:ss) 
    then you can use  datetime.fromisoformat to do operations on audiomoth and transponder system time
    """
    
    return datetime.strptime(time, "%Y%m%d_%H%M%S")

    
    
def make_timezone_aware(time):
    """
    Take a datetime object that is not timezone aware and make it aware of Zurich time zone
    """
    tz_zurich = pytz.timezone('Europe/Zurich')
    aware = time.replace(tzinfo=tz_zurich)
    return aware
def get_deployments(root, as_datetime = False):
    """
    return a list of deployments for which you have raw audio data, where each item in the list
    is a two item list with the first item the datetime of the beginning of the first day of the 
    recording and the second item the datetime of the end of the last day of the recording
    """
    
    #get the deployments
    deployments = load_json(os.path.join(root,'parameters', 'json_files','deployment_dates.json'))
    
    #convert to datetime
    if as_datetime:
        deployments = [[('_').join([i.split('-')[0], '000000']), ('_').join([i.split('-')[1], '235959'])] for i in deployments]
        deployments = [[audiomoth_to_datetime(i[0]), audiomoth_to_datetime(i[1])] for i in deployments]
    
    return deployments

def format_deployment(deployment):
    """
    Take an item from the output of get deployments and format it so it looks ike 'yyyymmdd-yyyymmdd'
    
    Arguments:
        deployment (list): 2 element list of start and end times for a deployment (ie, one item in the output of get_deployments)
        
    Returns:
        formatted_deployment (str): the deployment duration in the format 'yyyymmdd-yyyymmdd'
    """
    
    start = datetime_to_audiomoth(deployment[0]).split('_')[0]
    stop = datetime_to_audiomoth(deployment[1]).split('_')[0]
    formatted = ('-').join([start, stop])
    
    return formatted
def get_missing_days(save_dir):
    """
    get the days that are missing rfid data in the mouse stays table
    
    Arguments:
        save_dir (str): path to the directory containing the transponder readings (with one subdirectory for each audiomoth)
                        defaults to what this directroy should be

    """
    
    #check inputs
    assert os.path.exists(save_dir)
    
    missing_dict = {}
    moths = ['audiomoth00', 'audiomoth01', 'audiomoth02', 'audiomoth03', 'audiomoth04']

    missing_dates = []
    logs = [i for i in os.listdir(os.path.join(save_dir)) if not i.startswith('.')]

    for log in logs:
        print(log)

        #get which days you should have rfid data for
        box = log.split('_')[1]
        dates = log.split('_')[0]
        start = audiomoth_to_datetime(dates.split('-')[0]+'_000000')
        stop = audiomoth_to_datetime(dates.split('-')[1]+'_000000')
        delta = stop - start                       
        audio_days = [start + timedelta(days=i) for i in range(delta.days + 1)]
        audio_days = [i.date() for i in audio_days]
        print(sorted(audio_days))

        #get rfid data
        df = pd.read_csv(os.path.join(save_dir,log))
        rfid_days = list(set([datetime.fromisoformat(i).date() for i in df['event_time']]))
        print(sorted(rfid_days))
        print('\n')

        #check for missing dates
        for i in audio_days:

            if i not in rfid_days:

                missing_dates.append(str(i))

        #update dictionary
        missing_dict[dates] = {}
        missing_dict[dates][box] = missing_dates
        
    return missing_dict
def strptime_ms(val):
    "deal with datettime strings that have partial seconds. from https://stackoverflow.com/questions/3408494/string-to-datetime-with-fractional-seconds-on-google-app-engine"
    
    if '.' not in val:
        return datetime.strptime(val, "%Y-%m-%d %H:%M:%S")

    nofrag, frag = val.split(".")
    date = datetime.strptime(nofrag, "%Y-%m-%d %H:%M:%S")

    frag = frag[:3]  # truncate to milliseconds
    frag += (6 - len(frag)) * '0'  # add 0s
    return date.replace(microsecond=int(frag))
def time_correction(time, deployment, box, time_correction_df):
    """
    Take a timestamp from an audiomoth clock and use time_correction_df to correct it to the rfid clock
    """

    #convert to datetime
    time = strptime_ms(str(time))

    #get the time correction data
    row = time_correction_df[(time_correction_df['deployment'] == deployment) & (time_correction_df['box'] == box)]   
    assert len(row) == 1 #there must be exactly one row per box/deployment pair
    first_deployment_chime = list(row['first_deployment_chime'])[0]
    first_recovery_chime = list(row['first_recovery_chime'])[0]
    chime_total_time = list(row['recording_duration_on_audiomoth_clock'])[0]
    rfid_total_time = list(row['recording_duration_on_rfid_clock'])[0]
    deployment_correction = list(row['deployment_correction_seconds'])[0]
    recovery_correction = list(row['recovery_correction_seconds'])[0]
    
    # Define known relationship between audiomoth and rfid clocks (2 points)
    x = [0, chime_total_time] #audiomoth clock
    y = [deployment_correction, chime_total_time+recovery_correction] #rfid clock

    #get the line defined by these two points
    coefficients = np.polyfit(x, y, 1)
    linear_relationship = np.poly1d(coefficients)
    
    #get the elapsed time from the first annotated chime to this timestamp
    elapsed_time = (time - first_deployment_chime).total_seconds()
    
    #get the corrected time (ie, map this audiomoth timestamp to the rfid clock using linear_relationship)
    rfid_time = first_deployment_chime+timedelta(seconds = linear_relationship(elapsed_time))
    
    return rfid_time


def get_box_from_audiomoth(root, moth, deployment):
	"""
	Give a moth and deployment, get the box it recorded
	"""

	boxes_recorded = load_json(os.path.join(root, 'parameters','json_files', 'boxes_recorded.json'))

	try:
		box = boxes_recorded[moth][deployment]
		return box
	except:
		pass

def get_audiomoth_from_box(root, box,deployment):
	"""
	Give a box and deployment, get the audiomoth that recorded it
	"""

	boxes_recorded = load_json(os.path.join(root, 'parameters', 'json_files','boxes_recorded.json'))
	audiomoths = ['audiomoth00', 'audiomoth01', 'audiomoth02', 'audiomoth03', 'audiomoth04']

	#search for the box, deployment pair
	for moth in audiomoths:
		if deployment in boxes_recorded[moth].keys():
			if boxes_recorded[moth][deployment] == box:
				return moth
	return None
			
def get_boxes_from_deployment(root, deployment):
    """
    Give a deployment, get the boxes recorded during that deployment
    """
    
    #get the boxes recorded on this deployment
    boxes_recorded = load_json(os.path.join(root, 'parameters', 'json_files','boxes_recorded.json'))
    audiomoths = ['audiomoth00', 'audiomoth01', 'audiomoth02', 'audiomoth03', 'audiomoth04']
    boxes = [boxes_recorded[i][deployment] for i in audiomoths if deployment in boxes_recorded[i].keys()]
    return boxes

def get_dates_from_deployment(interval):
    start_date_str, end_date_str = interval.split('-')
    start_date = datetime.strptime(start_date_str, '%Y%m%d')
    end_date = datetime.strptime(end_date_str, '%Y%m%d')

    days = []
    current_date = start_date
    while current_date <= end_date:
        days.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    return days
def get_season_from_date(date):
	month = pd.to_datetime(date).month
	year = pd.to_datetime(date).year
	if month in [12, 1, 2]:
		return 'winter'
	elif month in [3, 4, 5]:
		return 'spring'
	elif (month in [6, 7, 8]) and year == 2022:
		return 'summer_22'
	elif (month in [6, 7, 8]) and year == 2023:
		return 'summer_23'
	elif (month in [6, 7, 8]) and (year != 2022) and (year != 2023):
		return 'summer'
	elif (month in [9, 10, 11]) and year == 2022:
		return 'autumn_22'
	elif (month in [9, 10, 11]) and year == 2023:
		return 'autumn_23'
	elif (month in [9, 10, 11]) and (year != 2022) and (year != 2023):
		return 'summer'
def merge_mouse_and_vocal_events(vocal_events, mouse_events, raw_counts):
    
    """
    
    Give mouse_events, vocal_events (with time correction), and raw_counts (with time correction) data frames
    Get a dataframe with one row per minue per box and counts/ids of cries, USVs, and mice
    
    """
    
    mv_events = []
    boxes_list = []
    deployments_list = []
    cry_count_list = []
    USV_count_list = []
    noise_count_list = []
    mouse_count_start_list = []
    mouse_count_end_list = []
    mouse_ids_start_list = []
    mouse_ids_end_list = []

    deployments = v_events['deployment'].unique()

    for deployment in tqdm(deployments): 
        #print(deployment)
        moths = v_events['moth'][v_events['deployment'] == deployment].unique()

        for moth in moths:
            #print('\t', moth)

            #get the vocal events for this moth and deployment
            these_v_events = v_events[(v_events['deployment'] == deployment) & (v_events['moth'] == moth)].reset_index(drop = True)

            #get the mouse events for this moth and deployment
            these_m_events = m_events[(m_events['deployment'] == deployment) & (m_events['audiomoth'] == moth)].reset_index(drop = True)

            #get the raw voc counts for this deployment and moth (which contains the beginning and end timestamps of each audio recording)
            this_recording = raw_counts[(raw_counts['deployment'] == deployment) & (raw_counts['moth'] == moth)]

            #define the start and end of the audio recording, ignoring the first 30 minutes and the last 30 minutes
            recording_start = this_recording['rfid_date'][31:-31].min()
            recording_end = this_recording['rfid_date'][31:-31].max()

            ### I think you have to convert these to rfid time

            #make 1 min intervals between start and end
            time_stamps = pd.date_range(start=recording_start, end=recording_end, freq='1T')

            for time_stamp in time_stamps: #for each time stamp

                #get the 55s interval corresponding to audio recording time
                start = time_stamp
                end = time_stamp + timedelta(seconds = 55)

                #subset vocal events to this interval (note we are operating only in rfid clock time)
                these_v_events = these_v_events[(these_v_events['rfid_start_seconds'] >= start) & (these_v_events['rfid_stop_seconds'] <= end)]

                #get the box
                boxes_list.append(box)

                #get the number of cries detected in the box in the interval
                cry_count_list.append(len(these_v_events['label'] == 'cry'))

                #get the number of USVs detected in the box in the interval
                USV_count_list.append(len(these_v_events['label'] == 'USV'))

                #get the number of noise events detected in the box in the interval
                noise_count_list.append(len(these_v_events['label'] == 'noise'))

                #get the number of mice in the box at the start of the interval

                #find the mouse event closest to the start of the interval
                closest_event_start = these_m_events.iloc[(these_m_events['event_time'] - start).abs().idxmin()]

                #find the closest mouse event t0 the end of the interval
                closest_event_end = these_m_events.iloc[(these_m_events['event_time'] - end).abs().idxmin()]

                if (closest_event_start['event_time'] - start) > timedelta(seconds=0): #if the closest event is after the start of the interval

                    #the mice in the box at the start of the interval is number in box at previous event
                    closest_event = these_m_events.iloc[(these_m_events['event_time'] - start).abs().idxmin() - 1]
                    mouse_count_start_list.append(len(ast.literal_eval(closest_event['occupant_ids_following_event'])))
                    mouse_ids_start_list.append(ast.literal_eval(closest_event['occupant_ids_following_event']))

                elif (closest_event_start['event_time'] - start) <= timedelta(seconds=0): #if the closest event is before the start of the interval

                    #the mice in the box at the start of the interval is number in box at this event
                    mouse_count_start_list.append(len(ast.literal_eval(closest_event['occupant_ids_following_event'])))
                    mouse_ids_start_list.append(ast.literal_eval(closest_event['occupant_ids_following_event']))

                if (closest_event_end['event_time'] - end) > timedelta(seconds=0): #if the closest event is after the end of the interval

                    #the mice in the box at the end of the interval is number in box at this event
                    closest_event = these_m_events.iloc[(these_m_events['event_time'] - end).abs().idxmin() - 1]
                    mouse_count_end_list.append(len(ast.literal_eval(closest_event['occupant_ids_following_event'])))
                    mouse_ids_end_list.append(ast.literal_eval(closest_event['occupant_ids_following_event']))

                elif (closest_event_end['event_time'] - end) <= timedelta(seconds=0): #if the closest event is before the end of the interval

                    #the mice in the box at the start of the interval is number in box at this event

                    mouse_count_end_list.append(len(ast.literal_eval(closest_event['occupant_ids_following_event'])))
                    mouse_ids_end_list.append(ast.literal_eval(closest_event['occupant_ids_following_event']))

                #make the dataframe

                these_mv_events = {
                    'deployment' : deployments_list,
                    'box': boxes_list,
                    'cry_count':cry_count_list,
                    'USV_count':USV_count_list,
                    'mouse_count_start':mouse_count_start_list,
                    'mouse_ids_start':mouse_ids_start_list,
                    'mouse_count_end':mouse_count_end_list,
                    'mouse_ids_end':mouse_ids_end_list
                }

                #append it
                mv_events.append(these_mv_events)

    return pd.concat(mv_events)
def generate_intervals(d1, d2):
    """
    give two timestamps (eg the start and end of a mouse stay)
    get a list of intervals that tile the time between those timestamps
    useful for calculating hourly box use
    """
    # if the stay was less than one hour, just return the start and stop time of the stay
    if (d2 - d1).total_seconds() < 60*60:
        return [(d1, d2)]
    
    else:
        d1_1 = d1.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        d2_1 = d2.replace(minute=0, second=0, microsecond=0)

        intervals = [
            (d1, d1_1),
            *((d1_1 + timedelta(hours=h), d1_1 + timedelta(hours=h+1)) for h in range((d2_1 - d1_1).seconds // 3600)),
            (d2_1, d2)
        ]

        return intervals
def contains_empty_wavs(path):
    """
    Give a full path to a directory of wav files from an audiomoth recording, get 1 if any wav file is zero bytes or 0 if not. 
    Useful for finding failed recordings.
    """
    
    # get the file names
    audio_timestamps = glob.glob(os.path.join(path, '*.WAV'))

    # find the files with 0 bytes
    zero_byte_files = [i for i in audio_timestamps if os.path.getsize(i) == 0]

    return len(zero_byte_files) > 1
                                 
def get_meetings_windows(meets):
    """
    Give a meets dataframe, get a dictionary indexed by box with the start and stop of each unique meet in that box.
    Useful for finding mouse meetings that overlap with recording windows from get_recording_windows
    """
    meets_dict = {}
    for box in meets['box'].unique():
        this_box = meets[meets['box'] == box]
        meets_dict[box] = [(start, stop) for start, stop in zip(this_box['overlap_start_time'].to_list(), this_box['overlap_end_time'].to_list())]
        
def find_overlapping_windows(listA, listB):
    
    """
    Give two lists of intervals, listA and listB. Get a list of the intervals in listA 
    that overlap with intervals in listB. Useful for finding mouse meetings that overlap with 
    recording windows.
    """
        
    overlaps =  [intervalA for intervalA in listA if any(
    (intervalA[0] <= intervalB[1] and intervalA[1] >= intervalB[0]) or
    (intervalB[0] <= intervalA[1] and intervalB[1] >= intervalA[0]
    ) for intervalB in listB)]
        
    return overlaps
def find_occupants(meets, box, time):
    """
    Give a meets dataframe, a box, and a time. Get a list of all the mice who were in the box at that time.
    """
#
#    # make sure times are in datetime
#    time = pd.to_datetime(time)
#    meets['overlap_start_time'] = pd.to_datetime(meets['overlap_start_time'])
#    meets['overlap_end_time'] = pd.to_datetime(meets['overlap_end_time'])

    # ignore meets that started 
    
    # subset the meets df
    these_meets = meets[(meets['box'] == box) 
                        & (meets['overlap_start_time'] <= time) 
                        & (meets['overlap_end_time'] >= time)]
    
    # get the occupant IDs
    id_list = pd.concat([these_meets['id1'], these_meets['id2']], ignore_index=True)
    
    # return the unique IDs
    return id_list.unique().tolist()

def get_deployment_from_time(root, time):
	"""
	Give a date, get a deployment string in 'yyyymmdd-yyyymmdd' format if that date falls in a deployment or nothing if not
	"""

	#convert
	time = pd.to_datetime(time)

	# get the deployments
	deployments = get_deployments(root)
	

	# check if the date is during a deployment 
	recorded_days = [pd.to_datetime(day) for day_list in [get_dates_from_deployment(i) for i in deployments] for day in day_list]

	if not pd.to_datetime(time.date()) in recorded_days:
		return 'not-recorded'

	else:
		# for each one, check if the date is between the start and end of the deployment
		starts = [i.split('-')[0] for i in deployments]
		stops = [i.split('-')[-1] for i in deployments]
		recorded = 0
		for start, stop in zip(starts, stops):
			
			if (pd.to_datetime(start) <= time) & (time < (pd.to_datetime(stop)+timedelta(days=1))):
				return ('-').join([start,stop])
			
		if recorded == 0:
			return 'not-recorded'
#      
def check_if_recorded(time, box):
	"""
	Give a time and box, get a 1 if that box was recorded with an AudioMoth at that time or 0 if not
	"""

	# get the deployment
	deployment = get_deployment_from_time(time)

	# if there is one
	if deployment == 'not-recorded':
		return 0

	elif deployment != 'not-recorded':

		# find the boxes
		boxes = get_boxes_from_deployment(deployment)

		# check if this box is one of them
		if box in boxes:
			return 1
		else: 
			return 0
def make_filename_timestamp():
    "Get a string of the date and time in yyyymmddhhmmss format. Useful for giving files and directories unique timestamps"
    
    now = str(datetime.now())
    
    return now.split(' ')[0].replace('-', '') + now.split(' ')[1].split('.')[0].replace(':', '')
def find_meets_stays_inconsistencies(meets, stays):
	"""
	Give a meets and a stays dataframe, get a list of mice in the meets df not in stays and vice versa. Useful for troubleshooting the
	igraph ValueError: Some vertices in the edge DataFrame are missing from vertices DataFrame
	"""
        
	stay_mice = stays['transponder_id'].unique()
	meet_mice = list(set(list(meets['id1']) + list(meets['id2'])))
	in_meets_but_not_stays = [i for i in meet_mice if i not in stay_mice]
	in_stays_but_not_meets = [i for i in stay_mice if i not in meet_mice]

	print('in stay_mice but not meet_mice\n')
	print(in_stays_but_not_meets)
	print('\n')
	print('in meet_mice but not stay_mice\n')
	print(in_meets_but_not_stays)
        
	return in_stays_but_not_meets, in_meets_but_not_stays
def calculate_night_duration(start_time, end_time, sunrises, sunsets):
    night_duration = timedelta(0)

    if start_time < sunrise:
        night_duration += min(sunrise, end_time) - start_time

    if end_time > sunset:
        night_duration += end_time - max(sunset, start_time)

    return night_duration

def calculate_stay_percent_night(row, sunrises, sunsets):
    entry_date = row['entry_time'].date()
    exit_date = row['exit_time'].date()

    # Ensure entry_date and exit_date exist in the dictionaries
    if entry_date not in sunrises or entry_date not in sunsets:
        return 0.0

    sunrise = sunrises[entry_date]
    sunset = sunsets[entry_date]

    start_time = pd.to_datetime(row['entry_time']).time()
    end_time = pd.to_datetime(row['exit_time']).time()

    night_duration = pd.Timedelta(0)

    if start_time < sunrise:
        night_duration += pd.Timestamp.combine(pd.Timestamp(entry_date), min(sunrise, end_time)) - pd.Timestamp.combine(pd.Timestamp(entry_date), start_time)

    if end_time > sunset:
        night_duration += pd.Timestamp.combine(pd.Timestamp(entry_date), end_time) - pd.Timestamp.combine(pd.Timestamp(entry_date), max(sunset, start_time))

    total_duration = row['exit_time'] - row['entry_time']
    percent_night = night_duration / total_duration if total_duration > pd.Timedelta(0) else 0

    return percent_night
def normalize_and_convert_date(date):
	"""
	Deal with text dates where you don't know the delimiter separating year/month/day and don't know if the format is yyyymmdd or ddmmyyyy
	"""

	# Normalize the delimiter
	normalized_date = '-'.join(re.split('[-.]', date))

	# Try parsing with dayfirst=True
	try:
		converted_date = pd.to_datetime(normalized_date, dayfirst=True)
		return converted_date
	except ValueError:
		pass

	# Try parsing with dayfirst=False
	try:
		converted_date = pd.to_datetime(normalized_date, dayfirst=False)
		return converted_date
	except ValueError:
		pass

	# If both attempts fail, return NaT
	return pd.NaT
def get_meets_by_hour(mouse, these_stays, these_meets, time_intervals):
	this_mouse_stays = these_stays[these_stays['transponder_id'] == mouse]
	this_mouse_meets = these_meets[(these_meets['id1'] == mouse) | (these_meets['id2'] == mouse)]
	result = []

	for _, row in this_mouse_stays.iterrows():
		entry_time = row['entry_time']
		exit_time = row['exit_time']
		stay = row['id']

		meet_data = []
		stay_intervals = generate_intervals(entry_time, exit_time)

		for i in stay_intervals:
			num_meets = len(this_mouse_meets[(this_mouse_meets['overlap_start_time'] < i[1]) & (this_mouse_meets['overlap_end_time'] > i[0])]) 
			round_interval_start = i[0].replace(minute=0, second=0, microsecond=0)
			round_interval_end = round_interval_start + timedelta(hours=1)
			hour_interval = str([round_interval_start.time(), round_interval_end.time()])
			meet_data.append({
				'transponder_id': mouse,
				'time': round_interval_start,
				'hour': hour_interval,
				'stay_id': stay,
				'num_meets': num_meets
			})

		mouse_data_df = pd.DataFrame(meet_data)
		for interval in time_intervals:
			if str(interval) in mouse_data_df['hour'].to_list():
				meets_this_hour = list(mouse_data_df['num_meets'][mouse_data_df['hour'] == str(interval)])[0]
			else:
				meets_this_hour = 0
			result.append({'transponder_id': mouse, 'hour': interval[0].hour, 'num_meets': meets_this_hour, 'stay_id':stay})
	
		#make a dataframe from the result list
		result_df = pd.DataFrame(result)
		result_df = result_df.groupby(['transponder_id', 'hour'])['num_meets'].sum().reset_index()

	return result_df
def get_recording_durations(root, save = False, save_dir = None, verbose = True):
	"""
	Get the timestamp of the first and last recording wav file for each deployment. Requires access to all raw data
	raw_wav_locations is a list of paths to the audio/raw directory in each of the lacie hard drives
	"""
    
	# initialize the dictionary to store everything
	recording_durations_dict = {}
	
	# get the storage locations
	recording_storage_locations = load_json(os.path.join(root, 'parameters', 'recording_storage_locations.json'))
	
	# for each deployment
	for deployment in recording_storage_locations.keys():
		
		if verbose:
			print(deployment)
		
		# check that you have access to the storage location
		assert os.path.exists(recording_storage_locations[deployment]), "Storage location not found - have you mounted the correct hard drives?"
		
		# get the audiomoths used
		audiomoths = os.listdir(recording_storage_locations[deployment])
		
		recording_durations_dict[deployment] = {}
		
		# for each audiomoth
		for moth in audiomoths:
			if verbose:
				print('\t', moth)
			
			recording_durations_dict[deployment][moth] = {}
			
			# get the box it recorded on this deployment
			box = get_box_from_audiomoth(root, moth, deployment)
			
			# now get the first and last timestamp from this recording, ignoring empty wav files
			
			# get the directory with the raw wav files for this deployment and box
			recording_dir = os.path.join(recording_storage_locations[deployment], moth, '_'.join([deployment, 'box'+str(box)]))
			
			if os.path.exists(recording_dir):
				
				# get the names of the wav files in this directory - these are the timestamps
				audio_timestamps = glob.glob(os.path.join(recording_dir, '*.wav'))
				assert len(audio_timestamps) > 0, "The raw wav directory has no files ending in .wav - are they .WAV instead and need to be fixed?"

				# convert filenames to datetime and sort them, ignoring the files that are empty/recorded after an audiomoth died
				audio_timestamps = sorted(audiomoth_to_datetime(os.path.split(i)[-1].split('.')[0]) for i in audio_timestamps if os.path.getsize(i) != 0)

				# get the first and last timestamp
				first_recording, last_recording  = min(audio_timestamps), max(audio_timestamps)

				# get the recording duration
				duration_seconds = (last_recording - first_recording).total_seconds()

				# add to dictionary
				recording_durations_dict[deployment][moth] = (str(first_recording), str(last_recording), duration_seconds)
	
	if save:
		save_json(recording_durations_dict, save_dir = save_dir, save_name = 'recording_durations.json')
		return recording_durations_dict
	
	else:
		return recording_durations_dict
def get_vocalizations_by_hour_of_day(season, v_events):
	# Set interval parameters
	interval_number = 24
	interval_duration = 24 / interval_number

	# Define intervals and labels
	intervals = pd.to_timedelta([interval_duration * i for i in range(interval_number + 1)], unit='h')
	labels = ['{}-{}'.format(i, i+1) for i in range(interval_number)]

	# Filter vocalization events for the given season
	df = v_events[v_events['season'] == season]

	# Assign vocalization times to intervals
	df['date'] = df['audiomoth_start_seconds'].dt.date
	df['combined_time'] = df['audiomoth_start_seconds'].dt.hour * 60 + df['audiomoth_start_seconds'].dt.minute
	df['interval'] = pd.cut(df['combined_time'], bins=intervals.total_seconds() / 60, labels=labels)

	# Count 'squeak' events by interval, deployment, and moth
	squeak_counts = df[df['label'] == 'squeak'].groupby(['date','deployment', 'moth', 'interval']).size().reset_index(name='squeak_count')

	# Count 'USV' events by interval, deployment, and moth
	USV_counts = df[df['label'] == 'USV'].groupby(['date','deployment', 'moth', 'interval']).size().reset_index(name='USV_count')

	# Merge counts
	all_counts = pd.merge(squeak_counts, USV_counts, on=['date','deployment', 'moth', 'interval'], how='outer')
	all_counts['hour'] = [int(i.split('-')[-1]) for i in all_counts['interval']]

	return all_counts

			 
			
			
			
		
    
    
    

    
    
    