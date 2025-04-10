#This file contains functions that help with using the AVA package from the Pearson Lab

import os
import h5py
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import random

from src import parameters, ava

def make_directories(ava_root, project_root):
    # make a directory structure for AVA inside the main project root
    # ie for each audiomoth and deployment, make a directory within whcih are features, projs, segs, and specs directories
    
    #list of the deployments
    deployment_dates_path = os.path.join(project_root, 'parameters', 'deployment_dates.json')
    
    #dict of the boxes recorded on each deployments
    boxes_recorded_path = os.path.join(project_root,'parameters', 'boxes_recorded.json')
    
    #get deployments
    deployments = parameters.load_json(deployment_dates_path) # this is a list

    #get the boxes
    boxes = parameters.load_json(boxes_recorded_path) # this is a nested dictionary

    #the audiomoths
    moths = ['audiomoth01', 'audiomoth02', 'audiomoth03', 'audiomoth04'] #ignore auduiomoth00

    #deployments that already have directories
    done = os.listdir(ava_root)

    #make an ava directory for each box/audiomoth

    #get the audiomoth and box
    for moth in moths:

        for deployment in deployments:

            if not (deployment in boxes[moth].keys()): #if this moth was not deployed on these dates
  
                continue

            else:

                #the box
                box = boxes[moth][deployment]

                #the name for the directory
                dir_name = ('_').join([deployment,moth,'box'+str(box)])

                if not (dir_name in done):
                    os.mkdir(os.path.join(ava_root, dir_name))
                    os.mkdir(os.path.join(ava_root, dir_name, 'features'))
                    os.mkdir(os.path.join(ava_root, dir_name, 'specs'))
                    os.mkdir(os.path.join(ava_root, dir_name, 'projs'))
                    os.mkdir(os.path.join(ava_root, dir_name, 'segs'))

                    if (dir_name in done) and not ('features' in os.listdir(os.path.join(ava_root, dir_name))):
                        os.mkdir(os.path.join(ava_root, deployment, 'features'))
                    elif (dir_name in done) and not ('specs' in os.listdir(os.path.join(ava_root, dir_name))):
                        os.mkdir(os.path.join(ava_root, deployment, 'specs'))
                    elif (dir_name in done) and not ('projs' in os.listdir(os.path.join(ava_root, dir_name))):
                        os.mkdir(os.path.join(ava_root, deployment, 'projs'))
                    elif (dir_name in done) and not ('segs' in os.listdir(os.path.join(ava_root, dir_name))):
                        os.mkdir(os.path.join(ava_root, deployment, 'segs'))

                    #make sure they're made       
                    assert os.listdir(os.path.join(ava_root, dir_name)) == ['features', 'specs', 'projs', 'segs']


    print('done.')
    
def get_audio_paths(ava_root, project_root, save=True):
	# define the paths to the raw audio and save them to a json


	#dict of the drive location (lacie or lacie2) for each raw audio
	recording_storage_locations_path = os.path.join(project_root, 'parameters', 'recording_storage_locations.json')

	recording_locations = parameters.load_json(recording_storage_locations_path)

	recordings = os.listdir(ava_root)

	deployments = list(set([i.split('_')[0] for i in recordings]))

	ava_audio_paths = {}

	for recording in recordings:

		deployment, moth, box = recording.split('_')[0], recording.split('_')[1], recording.split('_')[2]

		drive_location = recording_locations[deployment]

		if ('_').join([deployment, box]) in os.listdir(os.path.join(drive_location, moth)):

			path_to_audio = os.path.join(drive_location, moth, ('_').join([deployment, box]))
			ava_audio_paths[recording] = path_to_audio

	if save:

		parameters.save(ava_audio_paths, 
						save_dir = os.path.join(project_root, 'parameters'), 
						save_name = 'ava_audio_paths')  

	return ava_audio_paths

def write_segments(ava_root, project_root, model, features_root):
	#convert segments from das to ava and write to ava directory

	weird_names = ['20230915-20230917_audiomoth01_box4']

	#the ava directories
	recordings = [i for i in os.listdir(ava_root) if not i.startswith('.')]
	
	for recording in recordings:
		
		#recording info
		deployment = recording.split('_')[0]
		moth = recording.split('_')[1]
		box = recording.split('_')[2]
		
		#location of the segments csvs from deep audio segmenter
		features_file_name = '_'.join([deployment, moth, box, 'features.csv'])
		features_file_path = os.path.join(features_root, features_file_name)
		if os.path.exists(features_file_path): #if features exist for this deployment
			
			#location of the directory to write the segments text file in ava format
			ava_segments_dir = os.path.join(ava_root, recording, 'segs')
			assert os.path.exists(ava_segments_dir), f"Directory missing: {ava_segments_dir}"
			
			#location of the features
			ava_features_dir = os.path.join(ava_root, recording, 'features')
			features_file_path = glob.glob(os.path.join(ava_features_dir, '*features.csv'))[0]
			assert os.path.exists(features_file_path), f"Directory missing: {ava_segments_dir}"
			these_features = pd.read_csv(features_file_path)
			these_features = these_features[these_features['label'] != 'noise'] #ignore segments labeled as noise

			# Location of the raw audio
			raw_audio_dict = ava.get_audio_paths(ava_root=ava_root, project_root=project_root, save=False)
			assert len(list(these_features['wavs.dir'].unique())) == 1, "More than one unique wav directory found!"

			raw_audio_dir = these_features['wavs.dir'].iloc[0]
			assert os.path.exists(raw_audio_dir), f"Raw audio directory missing: {raw_audio_dir}"

			#list of available wav files
			all_wavs = sorted(glob.glob(os.path.join(raw_audio_dir, '*.wav')))
			
			# turn these names into segment files to check how many are done
			all_wavs_ava_names = [i.split('/')[-1].replace('.wav', '.txt') for i in all_wavs]

			# only keep the ones you haven't already done
			to_do_wavs = [i.replace('.txt', '.wav') for i in all_wavs_ava_names if not i in os.listdir(ava_segments_dir)]
			print(len(all_wavs_ava_names)-len(to_do_wavs), "are done out of", len(all_wavs_ava_names))
			
			#for each wav
			for wav in to_do_wavs:

				#find the segments in the features file if there are any
				these_segs = these_features[these_features['sound.files']==wav]

				if len(these_segs) == 0: #if there are no segments in this wav file, write an empty segments .txt

					ava_seg_name = wav.split('/')[-1].split('.')[0]+'.txt'
					file_path = os.path.join(ava_segments_dir, ava_seg_name)
					if not os.path.exists(file_path):
						with open(file_path, 'w') as f:
							lines_to_write = [(' ').join(['#Segments for', recording, wav])]
							for line in lines_to_write:
								f.write(line + '\n')

				else: #otherwise get the start and stop times for each of the segments, get the audio, and make spectrograms

					starts = these_segs['start'].to_list()

					#get the list to write
					lines_to_write = [(' ').join(['#Segments for', recording, wav])]

					for start in starts:
						stop = these_segs['end'][these_segs['start'] == start].iloc[0]
						lines_to_write.append(('\t').join([str(start), str(stop)]))

					assert len(lines_to_write) == len(starts) + 1, "You wrote fewer lines than you have segments. Something is wrong!"
					#get the file path
					ava_seg_name = wav.split('.')[0]+'.txt'
					file_path = os.path.join(ava_segments_dir, ava_seg_name)

					if not os.path.exists(file_path):

						# write the lines
						with open(file_path, 'w') as f:

							for line in lines_to_write:
								f.write(line + '\n')

			#check that all the das segment csvs now have an ava version                
			print(recording, 'is done...')

	print('all done.')
	
def inspect_hdf5(hdf5_file_path):
    """
    List all datasets in an HDF5 file.
    
    Parameters:
        hdf5_file_path (str): Path to the HDF5 file.
    
    Returns:
        list: A list of dataset paths.
    """
    datasets = []
    
    with h5py.File(hdf5_file_path, 'r') as f:
        def visit_func(name, node):
            if isinstance(node, h5py.Dataset):
                datasets.append(name)
        
        f.visititems(visit_func)
    
    return datasets

def count_audio_filenames_in_hdf5_files(directory):
	audio_counts = {}

	# Iterate over all HDF5 files in the directory
	for file_name in tqdm(os.listdir(directory)):
		if file_name.endswith('.hdf5'):
			file_path = os.path.join(directory, file_name)
			with h5py.File(file_path, 'r') as f:
				audio_filenames = f['audio_filenames'][:]

				# Count occurrences of each audio_filename
				for audio_filename in audio_filenames:

					audio_filename = audio_filename.decode('utf-8')  # Convert bytes to string if needed

					if audio_filename in audio_counts:
						audio_counts[audio_filename] += 1
					else:
						audio_counts[audio_filename] = 1

	return audio_counts


def check_ava_spec_completeness(directory, features_df):

	# Count occurrences of audio_filenames in HDF5 files
	hdf5_counts = count_audio_filenames_in_hdf5_files(directory)

	# Count occurrences of audio_filenames in features_df
	df_counts = features_df['full.path'].value_counts().to_dict()

	# Create a DataFrame for comparison
	comparison_data = []
	all_filenames = set(hdf5_counts.keys()).union(set(df_counts.keys()))
	for filename in all_filenames:
		hdf5_count = hdf5_counts.get(filename, 0)
		df_count = df_counts.get(filename, 0)
		comparison_data.append({
			'recording': directory.split('/')[-2],
			'audio_filename': filename,
			'hdf5_count': hdf5_count,
			'df_count': df_count,
			'match': hdf5_count == df_count
		})

	comparison_df = pd.DataFrame(comparison_data)
	
	return comparison_df

def sample_ava_spectrograms(ava_dir, n_samples):
    """
    Retrieve a sample of spectrograms from an ava recording	 directory.

    Parameters:
        root_dir (str): Root directory containing the animal directories.
        n_samples (int): Number of spectrograms to sample from each animal.

    Returns:
        list: A list of sampled spectrograms.
    """
    
    import h5py
    sampled_spectrograms = []

    # Iterate through each animal directory
    for recording_dir in os.listdir(ava_dir):
        recording_path = os.path.join(ava_dir, recording_dir, 'specs')

        if os.path.isdir(recording_path):
            # Get all HDF5 files in the 'specs' directory
            hdf5_files = [os.path.join(recording_path, f) for f in os.listdir(recording_path) if f.endswith('.hdf5')]

            for hdf5_file in hdf5_files:
                with h5py.File(hdf5_file, 'r') as f:
                    # Retrieve the spectrogram dataset
                    specs = f['specs'][:]
                    
                    # Sample spectrograms
                    num_specs = specs.shape[0]
                    sampled_indices = random.sample(range(num_specs), min(n_samples, num_specs))
                    sampled_spectrograms.extend([specs[i] for i in sampled_indices])

    return sampled_spectrograms
def sample_specs(ava_dir, n_samples, nth_recording):
    """
    Retrieve a sample of spectrograms from every nth animal's directory.

    Parameters:
        ava_dir (str): Root directory containing the animal directories.
        n_samples (int): Number of spectrograms to sample from each animal.
        nth_recording (int): Sample every nth recording directory.

    Returns:
        list: A list of sampled spectrograms.
    """
    
    try:
        import h5py
        assert h5py.__version__ is not None
        print("h5py is installed. Version:", h5py.__version__)
    except ImportError:
        assert False, "h5py is not installed"
    except AssertionError:
        assert False, "h5py is installed but version could not be determined"
    
    sampled_spectrograms = []

    # Get all recording directories
    recording_dirs = sorted([d for d in os.listdir(ava_dir) if os.path.isdir(os.path.join(ava_dir, d))])
    
    # Iterate through every nth animal directory
    for i, recording_dir in enumerate(recording_dirs):
        if i % nth_recording == 0:
            recording_path = os.path.join(ava_dir, recording_dir, 'specs')
            print(recording_dir)
            if os.path.isdir(recording_path):
                # Get all HDF5 files in the 'specs' directory
                hdf5_files = [os.path.join(recording_path, f) for f in os.listdir(recording_path) if f.endswith('.hdf5')]

                for hdf5_file in tqdm(hdf5_files):
                    with h5py.File(hdf5_file, 'r') as f:
                        # Check if 'specs' dataset exists
                        if 'specs' in f:
                            specs_dataset = f['specs']
                            num_specs = specs_dataset.shape[0]

                            # Sample spectrogram indices
                            sampled_indices = random.sample(range(num_specs), min(n_samples, num_specs))

                            # Read only the sampled spectrograms
                            for idx in sampled_indices:
                                sampled_spectrograms.append(specs_dataset[idx])

    return sampled_spectrograms


    
    