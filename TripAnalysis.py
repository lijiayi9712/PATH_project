import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dateutil import tz
import datetime
import os

filedate = '20170208'



def feet_to_km(number):
    return number/3280.8399

"""
=====================
trip length analysis
=====================
"""


def trip_size(file):
    plt.close()
    df = pd.read_csv(file)

    fig = plt.figure()  # Create matplotlib figure
    ax1 = fig.add_subplot(211)  # Create matplotlib axes
    ax2 = fig.add_subplot(212)  # Create another axes that shares the same x-axis as ax.

    ax1.hist(df['trip size'], cumulative=True, bins=100, density=1)  # np.arange(0, 40, 1)  bins = np.arange(0, 150, 1)
    ax1.set_xlabel('trip size (number of ∆ per trip)', fontsize=14)  # 'sample rate (number of sample points/minutes)'
    ax1.set_ylabel('count (number of trips)', fontsize=14)
    ax1.set_title('cdf_triplength(' + file[123:-12] + ')')
    ax1.xaxis.set_label_position("bottom")
    ax1.xaxis.set_label_coords(0.8, -0.18)
    ax1.titlesize: 14

    ax2.hist(df['trip size'], bins=100, density=1)  # np.arange(0, 40, 1)  bins = np.arange(0, 150, 1)
    ax2.set_xlabel('trip size (number of ∆ per trip)', fontsize=14)  # 'sample rate (number of sample points/minutes)'
    ax2.set_ylabel('count (number of trips)', fontsize=14)
    ax2.set_title('pdf_triplength(' + file[123:-12] + ')')
    ax2.xaxis.set_label_position("bottom")
    ax2.xaxis.set_label_coords(0.8, -0.18)
    ax2.titlesize: 14
    # ax2.set_yscale('log', nonposy='clip')
    fig.tight_layout()
    plt.savefig(directory + output_folder + 'trip_size (' + file[123:-12] + ')' + ".png")

"""
=============
Trip duration 
=============
"""
def trip_duration(file):
    plt.close()
    df = pd.read_csv(file)

    fig = plt.figure()  # Create matplotlib figure
    ax1 = fig.add_subplot(211)  # Create matplotlib axes
    ax2 = fig.add_subplot(212)  # Create another axes that shares the same x-axis as ax.

    ax1.hist(df['duration (in min)'], cumulative=True, bins=np.arange(0, 80, 0.5), density=1)  # np.arange(0, 40, 1)  bins = np.arange(0, 150, 1)
    ax1.set_xlabel('trip duration (in minutes)', fontsize=14)  # 'sample rate (number of sample points/minutes)'
    ax1.set_ylabel('count (number of trips)', fontsize=14)
    ax1.set_title('cdf_trip duration (' + file[123:-12] + ')')
    ax1.xaxis.set_label_position("bottom")
    ax1.xaxis.set_label_coords(0.8, -0.18)
    ax1.titlesize: 14

    ax2.hist(df['duration (in min)'], bins=np.arange(0, 80, 0.5), density=1)  # np.arange(0, 40, 1)  bins = np.arange(0, 150, 1)
    ax2.set_xlabel('trip duration (in minutes)', fontsize=14)  # 'sample rate (number of sample points/minutes)'
    ax2.set_ylabel('count (number of trips)', fontsize=14)
    ax2.set_title('pdf_trip duration (' + file[123:-12] + ')')
    ax2.xaxis.set_label_position("bottom")
    ax2.xaxis.set_label_coords(0.8, -0.18)
    ax2.titlesize: 14
    # ax2.set_yscale('log', nonposy='clip')
    fig.tight_layout()
    plt.savefig(directory + output_folder + 'trip_duration (' + file[123:-12] + ')' + ".png")







"""
========================
Total distance traveled
in the trip 
=======================
"""
def trip_distance(file):
    plt.close()
    df = pd.read_csv(file)

    fig = plt.figure()  # Create matplotlib figure
    ax1 = fig.add_subplot(211)  # Create matplotlib axes
    ax2 = fig.add_subplot(212)  # Create another axes that shares the same x-axis as ax.

    ax1.hist(df['distance'].apply(lambda x: x/3280.8399), cumulative=True, bins=np.arange(0, 500, 1), density=1)  # np.arange(0, 40, 1)  bins = np.arange(0, 150, 1)
    ax1.set_xlabel('trip distance (in km)', fontsize=14)  # 'sample rate (number of sample points/minutes)'
    ax1.set_ylabel('count (number of trips)', fontsize=14)
    ax1.set_title('cdf_trip distance (' + file[123:-12] + ')')
    ax1.xaxis.set_label_position("bottom")
    ax1.xaxis.set_label_coords(0.8, -0.18)
    ax1.titlesize: 14

    ax2.hist(df['distance'].apply(lambda x: x/3280.8399), bins=np.arange(0, 500, 1), density=1)  # np.arange(0, 40, 1)  bins = np.arange(0, 150, 1)
    ax2.set_xlabel('trip distance (in km)', fontsize=14)  # 'sample rate (number of sample points/minutes)'
    ax2.set_ylabel('count (number of trips)', fontsize=14)
    ax2.set_title('pdf_trip distance (' + file[123:-12] + ')')
    ax2.xaxis.set_label_position("bottom")
    ax2.xaxis.set_label_coords(0.8, -0.18)
    ax2.titlesize: 14
    # ax2.set_yscale('log', nonposy='clip')
    fig.tight_layout()
    plt.savefig(directory + output_folder + 'trip_distance (' + file[123:-12] + ')' + ".png")






"""
==================
median trip delta 
distribution
==================
"""



def tripdelta(file):
    plt.close()
    df = pd.read_csv(file)
    median_delta = df['median trip delta']
    median_delta.hist(bins=20)
    plt.xlabel('median trip delta', fontsize=14)  # 'sample rate (number of sample points/minutes)'
    plt.ylabel('count (number of trips)', fontsize=14)
    plt.title(' median trip delta(' + file[-32:-12] + ')')  # ' median sample rate('
    plt.titlesize: 18
    plt.show()


"""
==========================
trip hourly trend analysis
==========================
"""
# ===== convert timezones =====
utc = tz.tzutc()
pst = tz.gettz('America/Los Angeles')


def convert_time(timestamp):
    time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    time = time.replace(tzinfo=utc)
    return time.astimezone(pst)


def triphourly(file):
    plt.close()
    df = pd.read_csv(file)
    df['DateTime'] = [convert_time(d) for d in df["start time"]]
    df['Hour'] = [datetime.time(d).hour for d in df['DateTime']]
    df['Hour'].hist(cumulative=False, density=1, bins=np.arange(0, 25, 1))
    plt.xlabel('start hour', fontsize=14)
    plt.ylabel('count (number of trips)', fontsize=14)
    plt.title(' trip_hourly(' + filedate + ')' + '_' + str(minSIZE) + '_' + str(
        BARRIER))
    plt.titlesize: 18
    plt.show()

"""
=========================
difference between sample
time and system time
=========================
"""

def transmissionTime(file):
    plt.close()
    df = pd.read_csv(file)
    df['transmissionTime'] = (
                pd.to_datetime(df['SYSTEM_DATE'], format='%Y-%m-%d %H:%M:%S') - pd.to_datetime(df['SAMPLE_DATE'],
                                                                                               format='%Y-%m-%d %H:%M:%S')).dt.total_seconds()
    df['transmissionTime'].hist(bins=np.linspace(-500, 1500, 5))
    plt.xlabel('difference between system and sample time (in seconds)',
               fontsize=14)  # 'sample rate (number of sample points/minutes)'
    plt.ylabel('count (number of samples)', fontsize=14)
    plt.title('distribution of difference between system and sample time')
    plt.titlesize: 18
    plt.show()
    return df['transmissionTime'].describe()







#for min_travel_time in [120, 180, 240]:
 #   for min_median_speed in [5, 10, 15]:
 #       for min_travel_distance in [1000, 1800, 2400]:
 #           data_file1 = file_name_constructor(output_file_name="output_probe_data_I210.")
 #           data_file2 = file_name_constructor("CONSUMER", "output_probe_data_I210.")
 #           data_file3 = file_name_constructor("FLEET", "output_probe_data_I210.")
 #           trip_meta = file_name_constructor(output_file_name="trip_meta_I210.")
 #           sample_rate_subplots_histogram(data_file1)
  #          sample_delta_subplots_histogram(data_file2)
 #           sample_rate_subplots_histogram(data_file3)


min_travel_dist_list = [1000, 1800, 2400]
for min_travel_distance in min_travel_dist_list:
    min_travel_time = 180
    minSIZE = 5  # minimum number of timestamps needed to define a trip
    BARRIER = 240  # the maximum separation between consecutive timestamps within a trip (in seconds)
    #  minimum number of seconds for a trip to be considered valid
    min_median_speed = 5  # minimum median speed for a trip to be considered valid
    #min_travel_distance = 1000  # travel distance in feet, 1000ft to 0.5mi (2 or 3 blocks, Qijian asks)
    # (test sensitivity for min_travel_dist)

    directory = os.getcwd() + '/'

    folder = 'Feb_2017_DATA/data_raw/'

    output_folder = 'Feb_2017_DATA/data_raw/AnalysisOutput/' + filedate + '/'

    if not os.path.exists(directory + output_folder):
        try:
            os.makedirs(directory + output_folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


    def file_name_constructor(provider="", output_file_name=""):
        return directory + output_folder + output_file_name + filedate + '_' + str(minSIZE) + '_' + str(
            BARRIER) + '_' + str(min_travel_time) + '_' + str(min_median_speed) + '_' + str(
            min_travel_distance) + provider + ".waynep1.csv"


    trip_meta2 = file_name_constructor("CONSUMER", output_file_name="trip_meta_I210.")
    trip_meta3 = file_name_constructor("FLEET", output_file_name="trip_meta_I210.")
    input_with_trip_id = file_name_constructor(output_file_name="input_with_trip_id_I210.")

    trip_size(trip_meta2)
    trip_duration(trip_meta2)
    trip_distance(trip_meta2)

    trip_size(trip_meta3)
    trip_duration(trip_meta3)
    trip_distance(trip_meta3)


"""
corner cases analysis
probe_id = '3d87963d0a9b0720t2d3cb3d2674e9280'
df = pd.read_csv(data_file1)
sample_dates =  df.loc[df['PROBE_ID'] == probe_id]['SAMPLE_DATE']
sample_dates.sort_values()
length = df1.shape[0]

deltas = [1, 2, 2, 2, 3, 5, 5, 5, 5, 207]

df = pd.read_csv(trip_meta)
df.loc[df['probe id'] == probe_id]
"""
