# this file contains functions for pre-processing dataframes prior to anlaysis
import os
import feather
import pandas as pd
import numpy as np
from src.timestamps import is_recovery_date
import src.timestamps as tmp
from datetime import date, datetime, timedelta

from src.rfid import fk_id_to_box_id, box_id_to_fk_id, popcheck_id_to_database_id
from src.timestamps import check_sunup
from src.parameters import load_json

def raw_data_from_cloud_of_mice(df, df_type):
	"""
	Give a data frame of type df_type (stays, meets, box_events, vocal_segments, vocal_counts), get a stays df with the following pre-processing steps:
	1. Drop duplicates
	2. Drop 0 second stays or meets
	3. Add season
	4. Convert times to UTC
	5. Add columns for year, month, and day
	6. Remove test transponder from rfid tables (it's not a mouse)

	stays : stay table from cloud of mice
	meets: meet table from cloud of mice
	box_events: box_events tavle from cloud of mice
	vocal_segments: vocal segments from DAS inference (one row per vocalization with columns for vocal features)
	vocal_counts: vocalizations counted by minute from vocal_segments (one row per minute per box per audiomoth per deployment)
	"""

	#check inputs
	assert df_type in ['stays', 'meets', 'box_events', 'vocal_segments', 'vocal_counts', 'vocal_events'], "df_type must be 'stays', 'meets', 'box_events', 'vocal_counts', 'vocal_events'"

	#give some info
	print('Preprocessing a', df_type, 'table with', len(df), 'rows...')

	# preprocess
	if df_type == 'stays':
		subset=['id', 'entry_time', 'exit_time', 'transponder_id', 'time_in_secs', 'entry_id', 'exit_id']

		# deal with duplicates
		num_duplicates = df.duplicated(subset=subset).sum()
		if num_duplicates > 0:
			print("\t...found", num_duplicates, "duplicates")
			df= df.drop_duplicates(subset=subset)
			print("\t\t...dropped them.")

		# drop 0 length stays
		num_0_length = len(df[df['time_in_secs'] == 0])
		if num_0_length > 0:
			print("\t...found", num_0_length, "0s long stays")
			df = df[df['time_in_secs'] > 0]
			print("\t\t...dropped them.")

		# add seasons
		print('\t...adding season to stays')
		df['entry_time'] = pd.to_datetime(df['entry_time'])
		df['season'] = [tmp.get_season_from_date(i) for i in df['entry_time']]
		print('\t\t...done.')

		# convert to UTC
		print('\t...adding columns for stay start and end in UTC')
		df['entry_time'] = pd.to_datetime(df['entry_time'])
		df['exit_time'] = pd.to_datetime(df['exit_time'])
		df['entry_time_UTC'] = df['entry_time'].dt.tz_localize('Europe/Zurich', ambiguous = 'NaT').dt.tz_convert('UTC')
		df['exit_time_UTC'] = df['exit_time'].dt.tz_localize('Europe/Zurich', ambiguous = 'NaT').dt.tz_convert('UTC')
		print('\t\t...done.')

		# add year month and day columns
		print('\t...adding columns for year, month, day, and hour')
		df['year'] = df['entry_time'].dt.year
		df['month'] = df['entry_time'].dt.month
		df['hour'] = df['entry_time'].dt.hour
		print('\t\t...done.')

	elif df_type == 'meets':
		subset=['id', 'overlap_start_time', 'overlap_end_time']

		# deal with duplicates
		num_duplicates = df.duplicated(subset=subset).sum()
		if num_duplicates > 0:
			print("\t...found", num_duplicates, "duplicates")
			df= df.drop_duplicates(subset=subset)
			print("\t\t...dropped them.")

		# drop 0 length meets
		num_0_length = len(df[df['time_in_secs'] == 0])
		if num_0_length > 0:
			print("\t...found", num_0_length, "0s long meets")
			df = df[df['time_in_secs'] != 0]
			print("\t\t ...dropped them.")

		# drop closed boxes
		print('\t...dropping closed boxes')
		df = df[df['box'].isin(np.arange(2,41,2))]
		print('\t\t...done.')

		# add seasons
		print('\t...adding season to meets')
		df['overlap_start_time'] = pd.to_datetime(df['overlap_start_time'])
		df['season'] = [tmp.get_season_from_date(i) for i in df['overlap_start_time']]
		print('\t\t...done.')

		# convert to UTC
		print('\t...adding columns for overlap start and end in UTC')
		df['overlap_end_time'] = pd.to_datetime(df['overlap_end_time'])
		df['overlap_start_time_UTC'] = df['overlap_start_time'].dt.tz_localize('Europe/Zurich', ambiguous = 'NaT').dt.tz_convert('UTC')
		df['overlap_end_time_UTC'] = df['overlap_end_time'].dt.tz_localize('Europe/Zurich', ambiguous = 'NaT').dt.tz_convert('UTC')
		print('\t\t...done.')

		# add year month and day columns
		print('\t...additing columns for year, month, day, and hour')
		df['year'] = df['overlap_start_time'].dt.year
		df['month'] = df['overlap_start_time'].dt.month
		df['hour'] = df['overlap_start_time'].dt.hour
		print('\t\t...done.')

	elif df_type == 'box_events':
		subset = ['id','box_id', 'event_id', 'event_time']

		# deal with duplicates
		num_duplicates = df.duplicated(subset=subset).sum()
		if num_duplicates > 0:
			print("\t ...found", num_duplicates, "duplicates")
			df= df.drop_duplicates(subset=subset)
			print("\t\t...dropped them.")

		#rename columns
		print("\t...naming 'num_partners_after_event' to 'occupancy_after_event'")
		df = df.rename(columns={'num_partners_after_event':'occupancy_after_event'})
		print("\t\t...done.")

		#add box column
		print("\t...adding column for box")
		df['box'] = [fk_id_to_box_id(i) for i in df['box_id']]
		print("\t\t...done.")

		#add date
		print("\t...adding column for date")
		df['event_time'] = pd.to_datetime(df['event_time'])
		df['event_date'] = df['event_time'].dt.date
		print("\t\t...done.")

	elif df_type == 'vocal_segments': #TO DO this is redundant with vocal_events below
		subset=['label', 'audiomoth_start_seconds', 'audiomoth_stop_seconds']

		# deal with duplicates
		num_duplicates = df.duplicated(subset=subset).sum()
		if num_duplicates > 0:
			print("\t...found", num_duplicates, "duplicates")
			df= df.drop_duplicates(subset=subset)
			print("\t\t...dropped them.")

		# rename 'minute' to something more informative
		print("\t...renaming 'minute' column as 'timestamp' ...")
		df = df.rename(columns={'minute':'timestamp'})
		print("\t\t...done.")

		# add year month and day columns
		print('\t...adding columns for year, month, day, and hour')
		df['year'] = df['timestamp'].dt.year
		df['month'] = df['timestamp'].dt.month
		df['hour'] = df['timestamp'].dt.hour
		print('\t\t...done.')

		# add audiomoth timestamp
		print('\t...adding columns for year, month, day, and hour')
		df['audiomoth_timestamp'].fillna(df['date'], inplace = True)
		print('\t\t...done.')

#		# add a 'sunchange' column to indicate sunset and sunrise
#		print("\t...adding 'sunchange' column to indicate sunset (-1) and sunrise (1)")
#		df['sun_change'] = df['sunup'].diff()
#		print('\t\t...done.')

		# add season
		print('adding season to v_events...')
		v_events['audiomoth_start_seconds'] = pd.to_datetime(v_events['audiomoth_start_seconds'])
		v_events['season'] = ['summer_22' if (i.month in [6,7,8] and i.year == 2022) else 'summer_23' if (i.month in [6,7,8] and i.year == 2023) else 'autumn' if i.month in [9,10,11] else 'winter' if i.month in [12,1,2] else 'spring' for i in df['audiomoth_start_seconds']]
		print("\t\t..done.")

	elif df_type == 'vocal_counts':
		subset=['deployment', 'moth', 'box', 'audiomoth_timestamp']

		#rename cry to squeak
		if 'cry_count' in df.columns:
			print("\t...renaming 'cry' to squeak" )
			df = df.rename(columns={'cry_count':'squeak_count'})

		# add year month and day columns
		print('\t...additing columns for year, month, day, and hour')
		df['year'] = pd.to_datetime(df['audiomoth_timestamp']).dt.year
		df['month'] = pd.to_datetime(df['audiomoth_timestamp']).dt.month
		df['hour'] = pd.to_datetime(df['audiomoth_timestamp']).dt.hour
		print('\t\t...done.')

		# add a 'sunchange' column to indicate sunset and sunrise
		print("\t...adding 'sunchange' column to indicate sunset (-1) and sunrise (1)")
		df['sun_change'] = df['sunup'].diff()
		print('\t\t...done.')

		# add a column for total vocalizations regardless of type
		print("\t...adding column to count all vocalizations (squeak or USV)")
		df['vocalizations_count'] = df['squeak_count'] + df['USV_count']
		print('\t\t...done.')
		
		# add a column for hour start
		print("\t...adding column for 'hour_start'")
		df['hour_start'] = [(':').join([str(i).split(':')[0], '00', '00']) for i in df['hour']]
		print('\t\t...done.')
		
		# add season
		print('\t...adding season')
		df['audiomoth_timestamp'] = pd.to_datetime(df['audiomoth_timestamp'])
		df['season'] = [tmp.get_season_from_date(date) for date in pd.to_datetime(df['audiomoth_timestamp'])]
		print("\t\t..done.")
		
		# deal with duplicates
		num_duplicates = df.duplicated(subset=subset).sum()
		if num_duplicates > 0:
			print("\t...found", num_duplicates, "duplicates")
			df= df.drop_duplicates(subset=subset)
			print("\t\t...dropped them.")
			
	elif df_type == 'vocal_events':
		subset=['deployment', 'moth', 'box', 'audiomoth_start_seconds']
		
		# add year month and day columns
		print('\t...additing columns for year, month, day, and hour')
		df['year'] = pd.to_datetime(df['audiomoth_start_seconds']).dt.year
		df['month'] = pd.to_datetime(df['audiomoth_start_seconds']).dt.month
		df['hour'] = pd.to_datetime(df['audiomoth_start_seconds']).dt.hour
		print('\t\t...done.')

#		# add a 'sunchange' column to indicate sunset and sunrise
#		print("\t...adding 'sunup' and 'sunchange' column to indicate sunset (-1) and sunrise (1)")
#		df['sunup'] = [check_sunup(i) for i in df['audiomoth_start_seconds']]
#		df['sun_change'] = df['sunup'].diff()
#		print('\t\t...done.')

		# add a column for hour start
		print("\t...adding column for 'hour_start'")
		df['hour_start'] = [(':').join([str(i).split(':')[0], '00', '00']) for i in df['hour']]
		print('\t\t...done.')
		
		# add season
		print('\t...adding season')
		df['audiomoth_start_seconds'] = pd.to_datetime(df['audiomoth_start_seconds'])
		df['season'] = [tmp.get_season_from_date(date) for date in pd.to_datetime(df['audiomoth_start_seconds'])]
		print("\t\t..done.")
		
		#rename label
		if 'cry' in df['label'].unique():
			df['label'] = df['label'].map({'cry':'squeak', 'USV':'USV', 'noise':'noise'})
	
		# deal with duplicates
		num_duplicates = df.duplicated(subset=subset).sum()
		if num_duplicates > 0:
			print("\t...found", num_duplicates, "duplicates")
			df= df.drop_duplicates(subset=subset)
			print("\t\t...dropped them.")
		
	return df
			
	
        
def boxes_recorded(boxes_recorded_path):
	
	print('Getting boxes recorded...')
	boxes_recorded_json = load_json(boxes_recorded_path)
	df = pd.DataFrame(boxes_recorded_json)
	df = df.drop(columns = ['audiomoth00']) # drop audiomoth00 for now
	df = df.dropna().reset_index().rename(columns = {'index':'deployment'})
	df = df.sort_values(by = 'deployment').reset_index(drop = True)
	df['deployment_start'] = [pd.to_datetime(i.split('-')[0]) for i in df['deployment']]
	df['deployment_end'] = [pd.to_datetime(i.split('-')[1]) for i in df['deployment']]
	df['year'] = df['deployment_start'].dt.year
	df['season'] = df['deployment_start'].apply(lambda x: tmp.get_season_from_date(x))
	print('Done.')
	return df
import pandas as pd
def vocalization_by_time_of_day(season, v_events, stays):
    
	# Set interval parameters
    interval_number = 24
    interval_duration = 24 / interval_number

    # Define intervals and labels
    intervals = pd.to_timedelta(([interval_duration * i for i in range(interval_number)] + [24]), unit='h')
    labels = ['({}, {}]'.format(intervals[i-1], intervals[i]) for i in range(1, len(intervals))]

    # Filter vocalization events for the given season
    df = v_events[v_events['season'] == season]

    # Assign vocalization times to intervals
    df['combined_time'] = df['audiomoth_start_seconds'].dt.hour * 60 + df['audiomoth_start_seconds'].dt.minute
    df['audiomoth_start_time'] = df['audiomoth_start_seconds'].dt.time
    df['interval'] = pd.cut(df['combined_time'], bins=intervals.total_seconds() / 60, labels=labels)

    # Count 'squeak' events by interval
    squeak_counts = df[df['label'] == 'squeak'].groupby(['deployment', 'moth', 'interval']).size().reset_index(name='squeak_count')

    # Count 'USV' events by interval
    USV_counts = df[df['label'] == 'USV'].groupby(['deployment', 'moth', 'interval']).size().reset_index(name='USV_count')

    # Merge counts
    all_counts = squeak_counts.merge(USV_counts, on=['deployment', 'moth', 'interval'], how='outer').fillna(0)

    return all_counts

def convert_interbox_values_to_longform(distance_df, value_name='distance'):
    """
    Convert a square matrix DataFrame into a long-form DataFrame with unique pairwise combinations.
    
    Parameters:
    - distance_df: pd.DataFrame, square matrix DataFrame where columns and index are box numbers.
    - distance_col_name: str, name for the 'distance' column in the result.
    
    Returns:
    - pd.DataFrame: long-form DataFrame with columns for box_1, box_2, and distance.
    """
    # Melt the DataFrame to long format
    melted_df = distance_df.reset_index().melt(id_vars='index', var_name='box_2', value_name=value_name)
    melted_df.columns = ['box_1', 'box_2', value_name]

    # Ensure unique pairwise combinations ignoring order
    melted_df['min_box'] = np.minimum(melted_df['box_1'], melted_df['box_2'])
    melted_df['max_box'] = np.maximum(melted_df['box_1'], melted_df['box_2'])

    # Drop duplicates to keep only unique pairs
    unique_pairs_df = melted_df.drop(columns=['box_1', 'box_2']).drop_duplicates().rename(
        columns={'min_box': 'box_1', 'max_box': 'box_2'})

    # Reset the index for clarity
    unique_pairs_df.reset_index(drop=True, inplace=True)

    return unique_pairs_df
def normalize_and_convert_date(date):

    #deal with the delimiter
    normalized_date = '-'.join(re.split('[-.]', date))

    #try dayfirst=True
    try:
        converted_date = pd.to_datetime(normalized_date, dayfirst=True)
        return converted_date
    except ValueError:
        pass

    #try with dayfirst=False
    try:
        converted_date = pd.to_datetime(normalized_date, dayfirst=False)
        return converted_date
    except ValueError:
        pass

    #if both attempts fail, return NaT
    return pd.NaT




