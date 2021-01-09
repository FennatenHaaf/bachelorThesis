# -*- coding: utf-8 -*-
"""
This file contains extra functions for saving files, etc. 
that could be necessary in other classes

@author: Fenna ten Haaf
Written for the Econometrics & Operations Research Bachelor Thesis
Erasmus School of Economics
"""

import numpy as np
import os
from datetime import datetime
from datetime import timedelta 
from csv import writer
from tqdm import tqdm


#TODO some other time: create a print message function, with the option
# "verbose" so that I can choose not to print messages

def get_time():
    """ This function is to say what time it is at a certain point in
    the simulation
    """
    time = datetime.now()
    time_string = time.strftime("%H:%M:%S")   
    return time_string


def get_time_diff(start, end, time_format= '%H:%M:%S'):
    """ Gets the difference between two time strings""" 
    # convert from string to datetime objects
    start = datetime.strptime(start, time_format)
    end = datetime.strptime(end, time_format)
  
    if start <= end: # e.g., 10:33:26-11:15:49
        return end - start
    else: # end < start e.g., 23:55:00-00:25:00
        end += timedelta(1) # +day
        assert end > start
        return end - start
    
    
def get_standard_basket_dict(customer=True,basket=True,cat=True,
                             prod=True,price=True,week=True):
    """Returns the standard format for a dictionary for a datafile with
    baskets and products and customer ids"""
    data = {}
    data.setdefault("i", []) #customer
    data.setdefault("basket_hash", []) # basket number
    data.setdefault("c", []) #category
    data.setdefault("j", []) #product
    data.setdefault("price", []) #product price
    data.setdefault("t", []) # week
    
    return data


def save_df_to_csv(df, outdir, filename, add_time = True, add_index = False):
    """Saves a dataframe to a csv file in a specified directory, with a
    filename and attached to that the time at which the file is created"""
    
    if not os.path.exists(outdir):
            os.mkdir(outdir)
    
    if add_time:
        today = datetime.now()
        time_string = today.strftime("%Y-%m-%d_%Hh%mm")
  
        df.to_csv(f"{outdir}/{filename}_{time_string}.csv",index=add_index)
    else:
        df.to_csv(f"{outdir}/{filename}.csv",index=add_index)
    

#TODO: Fix this function
def write_to_csv(data, outdir, filename):
    """Function to write data into an existing csv file"""
    
    print("writing file to csv")
    #print(data)
    
    with open(f"{outdir}/{filename}.csv", 'a') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        data = np.array(data) # Turn into array
        
        for row in tqdm(range(len(np.array(data)))):
            #print(row)
            print(row)
            print(data[row])
            csv_writer.writerow(data[row])
        

