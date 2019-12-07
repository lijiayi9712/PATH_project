from preprocess import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
from datetime import datetime, timedelta
import seaborn as sns




def sample_rate_subplots_histogram(file, min_travel_time, min_median_speed, min_travel_distance):
    plt.close()
    df = pd.read_csv(file)
    if file[117:-12] == 'FLEET':
        bins = np.linspace(0, 10, 100)
        folder_cat = 'FLEET/'
    elif file[117:-12] == 'CONSUMER':
        bins = np.linspace(0, 30, 200)
        folder_cat = 'CONSUMER/'
    else:
        bins = np.linspace(0, 70, 300)
        folder_cat = 'OVERAL/'

    fig = plt.figure()  # Create matplotlib figure
    ax1 = fig.add_subplot(211)  # Create matplotlib axes
    ax2 = fig.add_subplot(212)  # Create another axes that shares the same x-axis as ax.

    ax1.hist(df['median_delta'].apply(lambda x: 60/x), cumulative=True, bins=np.linspace(0, 10, 100), density=1)  # np.arange(0, 40, 1)  bins = np.arange(0, 150, 1)
    ax1.set_xlabel('sample rate (in number of hits/minute)', fontsize=10)  # 'sample rate (number of sample points/minutes)'
    ax1.set_ylabel('count (number of probes)', fontsize=10)
    ax1.set_title('cdf' + ' sample rate(' + filedate + ')' + '_' + str(min_travel_time) + '_' + str(min_median_speed) + '_' + str(min_travel_distance))
    ax1.xaxis.set_label_position("bottom")
    ax1.xaxis.set_label_coords(0.8, -0.18)
    ax1.titlesize: 14

    ax2.hist(df['median_delta'].apply(lambda x: 60/x), bins=np.linspace(0, 10, 100))
    ax2.set_xlabel('sample rate (in number of hits/minute)', fontsize=10)
    ax2.set_ylabel('count (number of probes)', fontsize=10)
    ax2.set_title('pdf' + ' sample rate(' + filedate + ')' + '_' + str(min_travel_time) + '_' + str(min_median_speed) + '_' + str(min_travel_distance))
    ax2.xaxis.set_label_position("bottom")
    ax2.xaxis.set_label_coords(0.8, -0.18)
    ax2.titlesize: 14
    # ax2.set_yscale('log', nonposy='clip')
    fig.tight_layout()
    # ax1.set_xticks(fontsize=8, rotation=60)

    directory = os.getcwd() + '/'

    output_folder = 'Feb_2017_DATA/data_raw/AnalysisOutput/' + filedate + '/graphs/' + folder_cat

    if not os.path.exists(directory + output_folder):
        try:
            os.makedirs(directory + output_folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    plt.savefig(directory + output_folder + file[117:-12] + ".png")
    # plt.show()
    print('finished outputing graph')

def sample_delta_subplots_histogram(file):
    plt.close()
    df = pd.read_csv(file)

    fig = plt.figure()  # Create matplotlib figure
    ax1 = fig.add_subplot(211)  # Create matplotlib axes
    ax2 = fig.add_subplot(212) # Create another axes that shares the same x-axis as ax.

    ax1.hist(df['median_delta'], cumulative=True, bins = np.arange(0, 140, 0.5), density=1)   #np.arange(0, 40, 1)  bins = np.arange(0, 150, 1)
    ax1.set_xlabel('median sample delta (in seconds)', fontsize=10) #'sample rate (number of sample points/minutes)'
    ax1.set_ylabel('count (number of probes)', fontsize=10)
    ax1.set_title('cdf' + ' median sample delta(' + filedate + ')'+'_' + str(minSIZE) + '_' +str(BARRIER))
    ax1.xaxis.set_label_position("bottom")
    ax1.xaxis.set_label_coords(0.8, -0.18)
    ax1.titlesize: 14

    ax2.hist(df['median_delta'], bins = np.arange(0, 140, 0.5))
    ax2.set_xlabel('median sample delta (in seconds)', fontsize=10)
    ax2.set_ylabel('count (number of probes)', fontsize=10)
    ax2.set_title('pdf' + ' median sample delta(' + filedate + ')'+'_' + str(minSIZE) + '_' +str(BARRIER))
    ax2.xaxis.set_label_position("bottom")
    ax2.xaxis.set_label_coords(0.8, -0.18)
    ax2.titlesize: 14
    #ax2.set_yscale('log', nonposy='clip')
    fig.tight_layout()
    #ax1.set_xticks(fontsize=8, rotation=60)

    directory = os.getcwd() + '/'

    output_folder = 'Feb_2017_DATA/data_raw/AnalysisOutput/' + filedate + '/graphs/'

    if not os.path.exists(directory + output_folder):
        try:
            os.makedirs(directory + output_folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    plt.savefig(directory + output_folder + file[117:-12] + ".png")
    #plt.show()
    print('finished outputing graph')





filedate = '20170208'
directory = os.getcwd() + '/'
folder = 'Feb_2017_DATA/data_raw/'
output_folder = 'Feb_2017_DATA/data_raw/AnalysisOutput/' + filedate + '/'

if not os.path.exists(directory + output_folder):
    try:
        os.makedirs(directory + output_folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
minSIZE = 5
BARRIER = 240
min_travel_time = 180  # minimum number of seconds for a trip to be considered valid
min_median_speed = 5.0  # minimum median speed for a trip to be considered valid
min_travel_distance = 1000  # travel distance in feet, 1000ft to 0.5mi (2 or 3 blocks, Qijian asks)

def file_name_constructor(provider="", output_file_name=""):
    return directory + output_folder + output_file_name + filedate + '_' + str(minSIZE) + '_' + str(
        BARRIER) + '_' + str(min_travel_time) + '_' + str(min_median_speed) + '_' + str(min_travel_distance) + provider + ".waynep1.csv"


for min_travel_distance in [1000, 1800, 2400]:
    data_file1 = file_name_constructor(output_file_name="output_probe_data_I210.")
    data_file2 = file_name_constructor("CONSUMER", "output_probe_data_I210.")
    data_file3 = file_name_constructor("FLEET", "output_probe_data_I210.")
    trip_meta1 = file_name_constructor(output_file_name="trip_meta_I210.")
    trip_meta2 = file_name_constructor("CONSUMER", output_file_name="trip_meta_I210.")
    trip_meta3 = file_name_constructor("FLEET", output_file_name="trip_meta_I210.")
    sample_rate_subplots_histogram(data_file1, min_travel_time, min_median_speed, min_travel_distance)
    sample_rate_subplots_histogram(data_file2, min_travel_time, min_median_speed, min_travel_distance)
    sample_rate_subplots_histogram(data_file3, min_travel_time, min_median_speed, min_travel_distance)



data_file2 = file_name_constructor("CONSUMER", "output_probe_data_I210.")
data_file3 = file_name_constructor("FLEET", "output_probe_data_I210.")
sample_rate_subplots_histogram(data_file2, min_travel_time, min_median_speed, min_travel_distance)
sample_rate_subplots_histogram(data_file3, min_travel_time, min_median_speed, min_travel_distance)
