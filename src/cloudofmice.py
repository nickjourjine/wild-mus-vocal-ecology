#This file contains functions for accessing data from the uzh cloud of mice database
import os
import psycopg2
import getpass
import feather
import pandas as pd
import numpy as np
from src.parameters import load_json
from src.timestamps import is_recovery_date
import src.timestamps as tmp
from datetime import date, datetime, timedelta

from src.rfid import fk_id_to_box_id, box_id_to_fk_id, popcheck_id_to_database_id
from src.timestamps import check_sunup

def get_test_transponder(db_connection, deployment, save_dir, verbose, con):

	# This is the ID of the test transponder (converted to database format)
	testid = popcheck_id_to_database_id("000638D041")

	# Get times for query
	start, stop = get_query_timestamp_from_deployment(deployment=deployment)

	# Find when the test transponder was detected during the recording period (all boxes and times)
	df = test_transponder_query(con=db_connection, 
								start=start,
								end=stop,
								test_transponder_ID=testid,
								newserver=True,
								verbose=verbose, 
								close = False)


	# Set save directories
	save_name = f"{deployment}_test_transponder_readings.csv"

	# Save
	if not os.path.exists(os.path.join(save_dir, save_name)):
		if len(df) != 0:

			#add the box number
			active_fk_box_ids = load_json('/Volumes/LaCie_barn/mouse_barn_audiomoth/parameters/active_fk_box_ids.json')
			df = df.loc[df['fk_box'].isin(active_fk_box_ids)]
			df['box'] = [fk_id_to_box_id(i) for i in df['fk_box']]

			#add column indicating if chimes were generated at audiomoth recovery (1) or not (0)
			df['event_time'] = pd.to_datetime(df['event_time'])
			ymd = deployment.split('-')[0]
			time = ('-').join([ymd[:4], ymd[4:6], ymd[6:]])
			deployment_date = datetime.strptime(time, '%Y-%m-%d').date()
			df['recovery'] = [is_recovery_date(date, deployment) for date in df['event_time']]

			#save
			df.to_csv(os.path.join(save_dir, save_name), index=False)

		else: #save an empty df if no test transponder detected 
			df = pd.DataFrame(columns = ['id', 'event_time', 'antenna','line_hash', 'fk_transponder', 'fk_import_log', 'fk_box', 'box', 'recovery'])
			df.to_csv(os.path.join(save_dir, save_name), index=False)


def get_query_timestamp_from_deployment(deployment):

	assert ((len(deployment) == 17) and ('-' in deployment)), "Invalid deployment format. Please use 'yyyymmdd-yyyymmdd'."

	start, stop = deployment[:8], deployment[9:]

	# Get the start query
	start_query = f"{start[:4]}-{start[4:6]}-{start[6:8]} 00:00:00"

	# Get the stop query
	stop_query = f"{stop[:4]}-{stop[4:6]}-{stop[6:8]} 23:59:59"

	return start_query, stop_query

def get_box_occupancy(db_connection, deployment, savedir, lag=None, file_format = 'csv',verbose=False):
    # Query the box_activity database, where num_partners_after_event indicates the number of mice in the box
    # deployment (str): deployment in format 'yyyymmdd-yyyymmdd'
    # savedir (str): path of root directory for transponder CSVs

    # A list of all the boxes - use this loop if you want one file per box
    #boxes_list = list(range(2, 41, 2))
    #for box in boxes_list:

    assert file_format in ['csv', 'feather'], "file_format must be either 'csv' or 'feather'"

    if lag is not None:
        # Get the start time
        start, stop = deployment.split('-')

        # Subtract lag
        start = datetime.strptime(start, '%Y%m%d').date()
        start = start - timedelta(days=lag)
        start = start.strftime('%Y%m%d')
        dates = f"{start}-{stop}"
    else:
        start, stop = deployment.split('-')
        dates = deployment

    # Check if you have stay data for this month from this deployment
    if file_format == 'feather':
        save_name = f"{dates}_all_boxes_from_box_activity.feather"
    else:
        save_name = f"{dates}_all_boxes_from_box_activity.csv"

    #make a directory for the box events if you don't have one already
    if not os.path.exists(os.path.join(savedir, 'box_events')):
        os.mkdir(os.path.join(savedir, 'box_events'))

    if not os.path.exists(os.path.join(savedir, 'box_events', save_name)):

        # Convert dates to datetime
        start, stop = get_query_timestamp_from_deployment(deployment=dates)

        # Connect to the server and get all rows from all boxes during the deployment
        all_boxes = box_query(con=db_connection, cdate1=start, cdate2=stop, newserver=True, close=False, verbose=verbose)

        if file_format == 'feather':
            all_boxes.reset_index(inplace=True)
            all_boxes.to_feather(os.path.join(savedir, 'box_events', save_name))
        else:
            all_boxes.to_csv(os.path.join(savedir, 'box_events', save_name), index = False)

def get_mouse_stays(db_connection, deployment, savedir, lag = None, file_format = 'csv',verbose=False):
	# Use stay_query() to get mice in all boxes recorded during a given deployment and save CSVs
	# In order to properly count mice, always query starting from the earliest point prior to start in which box was empty

	# Arguments
	#   deployment (str): deployment in format 'yyyymmdd-yyyymmdd'
	#   savedir (str): path of root directory for transponder CSVs
	#   lag (int): lag value to subtract from the start date
	#   verbose (bool): if True stay_query() prints some info about access cloud of mice


	# Returns
	#   None

	# a list of all the boxes - use this loop if you want one file per box
#	boxes_list = list(range(2, 41, 2))
#	for box in boxes_list:

    assert file_format in ['csv', 'feather'], "file_format must be either 'csv' or 'feather'"

    if lag is not None:
        # get the start time
        start, stop = deployment.split('-')

        # subtract lag
        start = datetime.strptime(start, '%Y%m%d').date()
        start = start - timedelta(days=lag)
        start = start.strftime('%Y%m%d')
        dates = f"{start}-{stop}"
    else:
        start, stop = deployment.split('-')
        dates = deployment

    # check if you have stay data for this month from this deployment
    if file_format == 'feather':
        save_name = f"{dates}_all_boxes_from_mouse_stay.feather"
    else:
        save_name = f"{dates}_all_boxes_from_mouse_stay.csv"

    #make a directory for the mouse stays if you don't have one already
    if not os.path.exists(os.path.join(savedir, 'mouse_stays')):
        os.mkdir(os.path.join(savedir, 'mouse_stays'))

    if not os.path.exists(os.path.join(savedir, 'mouse_stays', save_name)):
        # convert dates to datetime
        start, stop = get_query_timestamp_from_deployment(deployment=dates)

        # get all rows from this box during the deployment, using min_event_time as cdate1
        # note that stay_query will return data starting with the max datetime prior to start at which the box was empty
        all_mice = stay_query(con=db_connection, cdate1=start, cdate2=stop, box=None, newserver=False, verbose=verbose, close=False)

        # save
        if file_format == 'feather':
            all_mice.reset_index(inplace=True)
            all_mice.to_feather(os.path.join(savedir, 'mouse_stays', save_name))
        else:
            all_mice.to_csv(os.path.join(savedir, 'mouse_stays', save_name), index=False)

def get_mouse_meetings(db_connection, deployment, savedir, lag=None, file_format = 'csv', verbose=False):
	# Query the mouse_meeting database for a specific deployment using the meetquery function
	# deployment (str): deployment in format 'yyyymmdd-yyyymmdd'
	# savedir (str): path of root directory for transponder csvs
	# lag (int): number of days prior to start of recording to look for mouse meetings
    
    assert file_format in ['csv', 'feather'], "file_format must be either 'csv' or 'feather'"

    # Get the start time name to check if the file exists
    if lag is not None:
            # get the start time
            start, stop = deployment.split('-')

            # subtract lag
            start = datetime.strptime(start, '%Y%m%d').date()
            start = start - timedelta(days=lag)
            start = start.strftime('%Y%m%d')
            dates = f"{start}-{stop}"
    else:
            start, stop = deployment.split('-')
            dates = deployment

    # Check if you have stay data for this month from this deployment
    if file_format == 'feather':
        save_name = f"{dates}_all_boxes_from_mouse_meeting.feather"
    else:
        save_name = f"{dates}_all_boxes_from_mouse_meeting.csv"

    #make a directory for the mouse meetings if you don't have one already
    if not os.path.exists(os.path.join(savedir, 'mouse_meets')):
            os.mkdir(os.path.join(savedir, 'mouse_meets'))

    # If it doesn't exist and the audiomoth was used in this deployment, make a file
    if not os.path.exists(os.path.join(savedir, "mouse_meets", save_name)):
        # Convert dates to datetime
        start, stop = get_query_timestamp_from_deployment(deployment=dates)

        # Connect to the server and get all rows with from all boxes during the period defined by dates
        all_meets = meet_query(con=db_connection, cdate1=start, cdate2=stop, newserver=True, close=False, verbose=verbose)

        if file_format == 'feather':
            all_meets.reset_index(inplace=True)
            all_meets.to_feather(os.path.join(savedir, 'mouse_meets', save_name))
        if file_format == 'csv':
            all_meets.to_csv(os.path.join(savedir, 'mouse_meets', save_name), index = False)

def stay_query(con=None, cdate1=None, cdate2=None, box=None, close=False, newserver=True, verbose=True, find_empty=False):
	"""
	Query the stays database.

	Parameters:
	- cdate1: str, optional. Date (or date/time) to start the query from.
	- cdate2: str, optional. Date (or date/time) to end the query.
	- boxnum: str, optional. The box of interest.
	- close: bool, optional. Close connection when done.
	- newserver: bool, optional. Use new server.
	- verbose: bool, optional. Report query and connection status.
	- find_empty: bool, optional. If True, search for the max time prior to cdate1 in which the box was empty.

	Returns:
	- stays: pandas DataFrame. Query results.
	"""
	# Perform database connection
	if not con:
		con = make_db_connection(verbose=verbose,newserver=newserver)
	cursor = con.cursor()

	# Get the fk_ID from the box number
	if box is not None:
		box_fk_ID = str(box_id_to_fk_id(box))

	if find_empty: # find the first time prior to the cdate1 where the box was empty 
		fullquery = f"""
			SELECT mouse_stay.*, transponder.transponder_id, transponder.sex, box.box_number
			FROM mouse_stay
			JOIN transponder ON mouse_stay.transponder_table_id = transponder.id
			JOIN box ON mouse_stay.box_id = box.id
			WHERE mouse_stay.box_id = '{box_fk_ID}'
			AND mouse_stay.exit_time >= (
				SELECT MAX(event_time)
				FROM box_activity
				WHERE box_activity.box_id = '{box_fk_ID}'
				AND box_activity.event_time <= '{cdate1}'
				AND box_activity.num_partners_after_event = 0
			)
			AND mouse_stay.entry_time <= '{cdate2}'
			ORDER BY mouse_stay.entry_time ASC
		"""
	# add WHERE mouse_stay.box_id = '{box_fk_ID}' if you want a specific box
	else: # or just all the complete stays that ended after cdate1 and started before cdate1
		fullquery = f"""
			SELECT mouse_stay.*, transponder.transponder_id, box.box_number
			FROM mouse_stay
			JOIN transponder ON mouse_stay.transponder_table_id = transponder.id
			JOIN box ON mouse_stay.box_id = box.id
			AND mouse_stay.exit_time >= '{cdate1}' 
			AND mouse_stay.entry_time <= '{cdate2}'
			ORDER BY mouse_stay.entry_time ASC
		"""

	if verbose:
		print("making query")
		print(fullquery)
	cursor.execute(fullquery)
	stays = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

	if close:
		if verbose:
			print("Closing connection")
		cursor.close()
		con.close()

	return stays

def box_query(con=None, cdate1=None, cdate2=None, box=None, close=False, newserver=True, verbose=True, find_empty=False):
	"""
	Query the box_activity database.

	Parameters:
	- cdate1: str, optional. Date (or date/time) to start the query from.
	- cdate2: str, optional. Date (or date/time) to end the query.
	- boxnum: str, optional. The box of interest.
	- close: bool, optional. Close connection when done.
	- newserver: bool, optional. Use new server.
	- verbose: bool, optional. Report query and connection status.
	- find_empty: bool, optional. If True, search for the max time prior to cdate1 in which the box was empty (useful for counting occupant ids).

	Returns:
	- box_activities: pandas DataFrame. Query results.
	"""
	# Perform database connection
	if not con:
		con = make_db_connection(verbose=verbose,newserver=newserver)
	cursor = con.cursor()


	# Get the fk_ID from the box number
	if box is not None:
		box_fk_ID = str(box_id_to_fk_id(box))

	if cdate1 is not None and cdate2 is not None:
		if find_empty: # find the first time prior to the cdate1 where the box was empty
			fullquery = f"""
				SELECT box_activity.*, mouse_event.event_type, transponder.transponder_id AS id1
				FROM box_activity
				JOIN mouse_event ON box_activity.event_id = mouse_event.id
				JOIN transponder ON mouse_event.transponder_table_id = transponder.id
				WHERE box_activity.box_id = '{box_fk_ID}'
				AND box_activity.event_time <= '{cdate2}'
				AND box_activity.event_time >= (
					SELECT MAX(event_time)
					FROM box_activity
					WHERE box_activity.box_id = '{box_fk_ID}'
					AND box_activity.event_time <= '{cdate1}'
					AND box_activity.num_partners_after_event = 0
				)
				ORDER BY box_activity.box_id, box_activity.event_time
			"""
		# add WHERE box_activity.box_id = '{box_fk_ID}' if you want a specific box
		else: # or just all the complete stays that ended after cdate1 and started before cdate1
			fullquery = f"""
				SELECT box_activity.*, mouse_event.event_type, transponder.transponder_id AS id1
				FROM box_activity
				JOIN mouse_event ON box_activity.event_id = mouse_event.id
				JOIN transponder ON mouse_event.transponder_table_id = transponder.id
				AND box_activity.event_time <= '{cdate2}'
				AND box_activity.event_time >= '{cdate1}'
				ORDER BY box_activity.box_id, box_activity.event_time
			"""

	cursor.execute(fullquery)
	box_activities = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

	if close:
		if verbose:
			print("Closing connection")
		cursor.close()
		con.close()

	return box_activities

def meet_query(con=None, cdate1=None, cdate2=None, close=False, newserver=True, verbose=True):
	"""
	Query the meetings database.

	Parameters:
	- cdate1: str, optional. Date (or date/time) to start the query from.
	- cdate2: str, optional. Date (or date/time) to end the query.
	- close: bool, optional. Close connection when done.
	- newserver: bool, optional. Use new server.
	- verbose: bool, optional. Report connection status.

	Returns:
	- meets2: pandas DataFrame. Query results.
	"""

	# Perform database connection
	if not con:
		con = make_df_connection(verbose=verbose,newserver=newserver)
	cursor = con.cursor()

	fullquery = f"""
		SELECT mouse_meeting.*, 
		T1.transponder_id AS id1, 
		T2.transponder_id AS id2,
		T1.sex AS sex1, 
		T2.sex AS sex2, 
		B1.box_number AS box,
		S1.entry_time AS entry1, S2.entry_time AS entry2,
		S1.exit_time AS exit1, S2.exit_time AS exit2,
		S1.time_in_secs AS length1, S2.time_in_secs AS length2
		FROM mouse_meeting
		JOIN transponder AS T1 ON mouse_meeting.transponder_table1_id = T1.id
		JOIN transponder AS T2 ON mouse_meeting.transponder_table2_id = T2.id
		JOIN mouse_stay AS S1 ON mouse_meeting.stay1_id = S1.id
		JOIN mouse_stay AS S2 ON mouse_meeting.stay2_id = S2.id
		JOIN box AS B1 ON mouse_meeting.box_id = B1.id
		WHERE mouse_meeting.overlap_end_time >= '{cdate1}' AND mouse_meeting.overlap_start_time <= '{cdate2}'
		ORDER BY mouse_meeting.overlap_start_time ASC

	"""


	if verbose:
		print("making query")
		print(fullquery)

	cursor.execute(fullquery)
	meets = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

	if close:
		if verbose:
			print("closing connection")
		cursor.close()
		con.close()

	return meets

def test_transponder_query(con=None, start=None, end=None, test_transponder_ID=None, newserver=True, verbose=True, close=False):
	"""
	Query the readings database to find when the test transponder was read.

	Parameters:
	- con: the connection to the database
	- start: str, optional. Start date (or date/time) to start the query from in the format 'yyyy-mm-dd hh:mm:ss'.
	- end: str, optional. End date (or date/time) to end the query in the format 'yyyy-mm-dd hh:mm:ss'.
	- test_transponder_ID: str, optional. The ID of the test transponder converted format, e.g., '00-06-38-d0-41'.
	- newserver: bool, optional. Use new server.
	- verbose: bool, optional. Report query and connection status.
	- close: if True, close the connection

	Returns:
	- data: pandas DataFrame. Query results.
	"""
	if not con:
		con = make_db_connection(verbose=verbose,newserver=newserver)
	cursor = con.cursor()

	query = f"""SELECT * FROM reading
				WHERE reading.fk_transponder = (SELECT id FROM transponder WHERE transponder_id = '{test_transponder_ID}')
				AND reading.event_time <= '{end}'
				AND reading.event_time >= '{start}'"""

	if verbose:
		print("Making query:")
		print(query)

	cursor.execute(query)
	data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

	# Close the connection
	if close:
		cursor.close()
		con.close()

	return data


def make_db_connection(verbose=True, newserver=True):
	if newserver:
		host = "ieu-clom.bioc.uzh.ch"
	else:
		host = '130.60.168.212'

	# Check if connection already exists and is valid
	new_connect = False
	if 'con' in globals():
		if not con.closed:
			new_connect = True
	else:
		new_connect = True

	if new_connect:
		if verbose:
			print(f"Opening new connection {host}")

		# Prompt the user to enter the password
		password = getpass.getpass("Enter the password for mouse_user: ")

		# Establish a new connection
		con = psycopg2.connect(database="antenna_data", user="mouse_user", password=password, host=host, port="5432")

	return con


def kill_db_connections(verbose=True):
	if 'con' in globals():
		if con.closed == 0:
			if verbose:
				print("closing connection")
			con.close()
		elif verbose:
			print("no connection to close")
	elif verbose:
		print("no connection to close")
        
def find_box_rfid_gaps(db_connection, start_date, end_date, box, min_gap_hours, save_dir=None, save=True, verbose=False):
	# Convert dates
	start_date = datetime.strptime(start_date + ' 00:00:00', '%Y%m%d %H:%M:%S')
	end_date = datetime.strptime(end_date + ' 23:59:59', '%Y%m%d %H:%M:%S')

	# Query box_activity
	activity = box_query(con= db_connection, cdate1=start_date, cdate2=end_date, box=box, verbose=verbose, find_empty=False)

	# Convert the event_time column to datetime
	activity['event_time'] = pd.to_datetime(activity['event_time'])

	# Calculate the time differences between consecutive timestamps
	time_diffs = activity['event_time'].diff()

	# Get indices of gaps longer than min_gap_hours
	gap_indices = time_diffs[time_diffs > timedelta(hours=12)].index

	# Initialize a DataFrame to store gap information
	gap_df = pd.DataFrame(columns=['date', 'box', 'gap_start', 'gap_stop', 'gap_length_hours'])

	# Iterate over gap indices and extract gap start and stop times
	for i in gap_indices:
		gap_stop = activity.loc[i, 'event_time']
		gap_start = activity.loc[i-1, 'event_time']
		gap_length = (gap_stop - gap_start).total_seconds()/3600
		day_of_gap_start = gap_start.strftime('%Y-%m-%d')

		# Create a temporary DataFrame for the current gap
		temp_df = pd.DataFrame({'date': [day_of_gap_start],
								'box': [box],
								'gap_start': [gap_start],
								'gap_stop': [gap_stop],
								'gap_length_hours': [gap_length]})

		# Concatenate the temporary DataFrame with the main DataFrame
		gap_df = pd.concat([gap_df, temp_df], ignore_index=True)


	if save:
		save_name = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}_box{box}_gaphrs{min_gap_hours}_rfid_reading_gaps.csv"
		gap_df.to_csv(os.path.join(save_dir, save_name), index=False)
	else:
		return gap_df

def check_stays(df, time_delta):
    
    """Check if any consecutive exit times in a stays datafram differ by more than a specified time delta.

    Args:
        df (pandas.DataFrame): The DataFrame containing exit times.
        time_delta (float): The maximum allowed time difference between consecutive exit times in hours.

    Returns:
        Tuple[List[float], Optional[str]]: A tuple containing:
            - time_diffs_seconds (List[float]): A list of time differences in seconds between consecutive exit times.
            - warning_message (Optional[str]): A warning message indicating if any time difference exceeds the specified time delta. 
              Returns None if all time differences are within the threshold."""
    
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    
    # Sort the DataFrame by exit_time
    sorted_df = df.sort_values(by='exit_time')
    
    # Calculate time differences between consecutive exit times
    time_diffs = sorted_df['exit_time'].diff().dropna()
    
    # Find pairs of dates and their corresponding row indices with time differences larger than time_delta
    pairs = []
    for idx, diff in time_diffs.iteritems():
        if diff.total_seconds() > time_delta * 3600:
            exit_time1 = sorted_df.loc[idx - 1, 'exit_time']
            exit_time2 = sorted_df.loc[idx, 'exit_time']
            pairs.append((str(exit_time1), str(exit_time2), idx - 1, idx))
    
    # Check if any time difference is larger than time_delta hours
    if any(diff.total_seconds() > time_delta * 3600 for diff in time_diffs):
        warning_message = "Warning: One or more row-to-row time differences larger than {} hours.".format(time_delta)
    else:
        warning_message = None
    
    return pairs, warning_message


    
    
    
