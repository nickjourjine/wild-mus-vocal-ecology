#this file contains funcitons for getting weather data 

import pandas as pd
from datetime import datetime
from meteostat import Point, Hourly
from meteostat import Stations


def get_stations(latitude, longitude, verbose = True):
    """
    Get the closest weather station to given pair of GPS coordinates and return how far it is from those coordinates.
    
    Return the station data. If verbose is True, print the station name and distance to the provided coordinates.
    """

    # Get nearby weather stations
    stations = Stations() #intialize 
    stations = stations.nearby(latitude, longitude) #get nearby stations
    station = stations.fetch(1) #return the closest one (change 1 number to n to n closest)
    
    #distance in km
    distance = station['distance'][0]/1000
 
    return station
    
def get_weather(latitude, longitude, start, end):
    """
    Use meteostat to get weather records between start and end from the station closest to [latitude,longitude]
    """
    
    #check the inputs
    assert isinstance(latitude, float)
    assert isinstance(longitude, float)
    assert isinstance(start, datetime)
    assert isinstance(end, datetime)
    
    #get the nearest weather station to the provided coordinates
    station = get_stations(latitude, longitude)
    station_lat, station_lon = station['latitude'][0], station['longitude'][0]
    
    #create a point for this station
    location = Point(station_lat, station_lon)
    
    #get the hourly data from this station between start and end
    data = Hourly(location, start, end)
    data = data.normalize() #fill gaps with NaN
    data = data.interpolate(limit=10000) #interpolate between gaps (turn NaNs into 0)
    data = data.fetch()
    data['minute'] = data.index
    data = data.reset_index(drop=True)
    
    #make minutely so you can merge with voc counts easily
    dfs = []
    for i in data.index:
        hour = pd.DataFrame(data.iloc[i]).transpose()
        minutely = pd.concat([hour]*60, ignore_index=True)
        minutely['minute'] = pd.date_range(minutely['minute'][0], periods=60, freq='min').tolist()
        dfs.append(minutely)
    
    weather_df = pd.concat(dfs, ignore_index=True)
    
    return weather_df

def translate_weather_code(code):
    """
    take a weather condition code and return the weather description from https://dev.meteostat.net/python/#data-structure
    """
    
    #the dictionary for translating codes to descriptions
    code_dict = {1: 'Clear', 
                 2: 'Fair', 
                 3:'Cloudy', 
                 4:'Overcast', 
                 5:'Fog', 
                 6:'Freezing Fog', 
                 7:'Light Rain', 
                 8:'Rain', 
                 9:'Heavy Rain', 
                 10:'Freezing Rain', 
                 11:'Heavy Freezing Rain', 
                 12:'Sleet', 
                 13:'Heavy Sleet', 
                 14:'Light Snowfall', 
                 15:'Snowfall', 
                 16:'Heavy Snowfall', 
                 17:'Rain Shower', 
                 18:'Heavy Rain Shower', 
                 19:'Sleet Shower', 
                 20:'Heavy Sleet Shower', 
                 21:'Snow Shower', 
                 22:'Heavy Snow Shower', 
                 23:'Lightning', 
                 24:'Hail', 
                 25:'Thunderstorm', 
                 26:'Heavy Thunderstorm', 
                 27:'Storm'
                }
    
    #translate and return
    weather = code_dict[code]
    
    return weather