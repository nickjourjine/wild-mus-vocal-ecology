#this file contains functions for modifying, listing, and sorting files and file paths
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import glob
import re
import os



def fix_extensions(wavs_dir):
    """
    change .WAV to .wav in raw audiomoth directories
    only have to do this once

    Parameters
    ----------
    wavs_dir (str): full path to audiomoth directory from a given deployment with raw recorings

    Return
    ------
    None

    """

    to_fix = [i for i in os.listdir(wavs_dir) if i.endswith('.WAV') and not i.startswith('.')]
    if len(to_fix) > 0 :
        example_old_name = to_fix[3]
        example_new_name = example_old_name.split('.WAV')[0]+'.wav'

        print('example old name:', example_old_name)
        print('example new name:', example_new_name)
        #confirm = input('An example of the change is above - does this look ok? If y all files will be changed (y/n)')
        #if confirm == 'y':

        for old_name in to_fix:
            new_name = old_name.split('.WAV')[0]+'.wav'
            os.rename(os.path.join(wavs_dir,old_name), os.path.join(wavs_dir,new_name))

        print('changed all file extensions...')
            
#        else:
#            print('ok, take a look at the file names and try again....')
#            return

    else:    
        return

#sort files by time - from https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/

#related to sorting
def sort_nicely(to_sort):
    """ Sort the given list in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    to_sort.sort( key=alphanum_key )
    return to_sort

#related to handling dataframes
def combine_dataframes(source_dir, save_dir, save_name, file_format, include_string, exclude_string, paths_list=None, save=False):

    """
    combine multiple csvs from a single directory (default) or list of paths into one. 

    Parameters
    ----------
    paths_list (list): a list of full paths to the csvs to be combined

    source_dir (string): the path to the directory containing the annotation csvs to be combined

    save_dir (string): the path to the directory where the combined csv will be saved

    save_name (string): name of the combined csv to be saved

    file_format (string): '.csv' or '.feather'

    include_sting (string): only combine files with this string in their name

    exclude_string (string): ignore files with this in their name
    
    save (boolean): if True, save the csv. If False, just return the dataframe

    Returns
    -------
    all_files (dataframe): the combined dataframe

    """

    if paths_list == None and source_dir != None:
        sources = [os.path.join(source_dir,i) for i in os.listdir(source_dir) if i.endswith(file_format) and exclude_string not in i and not i.startswith('.') and include_string in i]
        combined = []

    elif paths_list != None and source_dir == None:
        sources=paths_list
        combined=[]

    elif paths_list == None and source_dir == None:
        print('provide either a list of paths or a directory containing all the files to be combined')

    elif paths_list != None and source_dir != None:
        print('provide either a list of paths or a directory containing all the files to be combined, not both')

    if file_format == '.csv':

        for i in sources:
            temp = pd.read_csv(i)
            if len(i) != 0:
                combined.append(temp)
            else:
                print(i, 'is empty')

        all_files = pd.concat(combined)
        if save:
            all_files.to_csv(os.path.join(save_dir,save_name)+'.csv', index=False)
        return all_files

    elif file_format == '.feather':

        for i in sources:
            temp = pd.read_feather(i)
            if len(i) != 0:
                combined.append(temp)
            else:
                print(i, 'is empty')

        all_files = pd.concat(combined)
        all_files = all_files.reset_index(drop=True)
        if save:
            all_files.to_feather(save_dir+save_name+'.feather')
        return all_files


    
    
#related to saving/loading json files
def get_paths_raw(raw_root_dir):
    """
    get paths to all of the raw recordings 

    wav_root_dir (str): full path to the directory where the the raw recordings are stored
    bg_root_dir (str): full path to the directory where the the background examples for each deployment are stored
    return a dictionary of paths indexed by audiomoth ID
    """

    paths_dict = {}
    moths_list = os.listdir(raw_root_dir)
        
    for moth in moths_list:
        paths_dict[moth] = glob.glob(os.path.join(raw_root_dir,moth+'/*')+'/')
        
    return paths_dict

# data visualization/plotting

#related to plotting/visualization

