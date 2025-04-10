#this file contains functions for saving, loading, and keeping track of parameters for different analyses
import os
import json
from datetime import datetime

def save_json(params, save_dir, save_name):
    """
    save a dictionary as .json and double check if you want to overwrite it.

    Parameters
    ----------
    params_dict (dict): the parametes dictionary to be saved

    save_dir (str): the path to the place where the dictionary file will be saved

    save_name (str): the name of the file without any file extension

    Returns
    -------
    None

    """

    save_path = os.path.join(save_dir,save_name)

    if save_name not in os.listdir(save_dir):
        with open(save_path, 'w') as fp:
            json.dump(params, fp, indent=4)

        print('no file with save_name existed in save_dir...saved the params')
        return

    else: 
        print('This file already exists in save_dir:', save_name)
        val = input('overwrite? y/n')

        if val == 'y':
            val = input('are you sure? y/n')

            if val == 'y':
                with open(save_path, 'w') as fp:
                    json.dump(params, fp, indent=4)
                print('ok - replaced existing file')
                return

            elif val == 'n':
                print('ok - no file saved')
                return

        elif val == 'n':
                print('ok - no file saved')
                return

        else:
                print('no file saved...')
                return

    return


def load_json(save_path):
    """
    load a dictionary from .json 

    Parameters
    ----------

    path(str):path to the file

    Returns
    -------
    params_dict (dict): the params dictionary you saved

    """

    with open(save_path, 'r') as fp:
            params_dict = json.load(fp)

    return params_dict

#related to file paths
def get_timestamp():
    """
    uses datetime to return a string with the format CurrentDate_CurrentTime (#####_#####)
    useful for naming directories

    Parameters
    ----------
    None

    Return
    ------
    The date and time as a string, e.g. 20220920_120000

    """
    
    current_date = str(datetime.now()).split(' ')[0].replace('-', '') 
    current_time = str(datetime.now()).split(' ')[1].split('.')[0].replace(':', '')

    return ('_').join([current_date, current_time])

def save_model(model, data, save_dir, training_params, timestamp):
    """
    Save a random forest model and its training parameters
    
    Arguments:
        model (RandomForest object): the model to save
        save_dir (str): the directory where the model and parameters will be save
        
    Returns:
        None
        
    """

    params_save_name = ('_').join([timestamp, 'training_parameters.json'])
    model_save_name
    
    print('model and training parameters will be saved to...', params_save_dir)
    response = input('continue? y/n')

    if response == 'n':
        print('ok - doing nothing')
        return

    elif response == 'y':

        #save the parameters
        parameters.save(params = model_params,
                        save_dir = params_save_dir, 
                        save_name = params_save_name)

        #save the model
        model_save_name = ('_').join([model_type,iteration,'voc_type_model'])   
        pickle.dump(model, open(os.path.join(params_save_dir,model_save_name)+'.pkl', 'wb'))

        #make sure you actually saved
        assert os.path.exists(os.path.join(params_save_dir,model_save_name)+'.pkl')

        print('saved model to:\n\t', os.path.join(params_save_dir,model_save_name)+'.pkl')
        print('done.')

    
    
    
    
    