import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dateutil import tz
from datetime import datetime, timedelta
import os, sys

from utils import *

"""
=====================
trip length analysis
=====================
"""
def feet_to_km(number):
    return number/3280.8399

def trip_size(df, title, output_path):
    plt.close()
    fig = plt.figure()  # Create matplotlib figure
    ax1 = fig.add_subplot(211)  # Create matplotlib axes
    ax2 = fig.add_subplot(212)  # Create another axes that shares the same x-axis as ax.

    ax1.hist(df['trip size'], cumulative=True, bins=np.arange(0,500,1), density=1) 
    ax1.set_xlim(left=0, right=500)
    ax1.set_xlabel('trip size (number of ∆ per trip)', fontsize=14)  
    ax1.set_ylabel('count (number of trips)', fontsize=14)
    ax1.set_title('cdf_tripsize(' + title + ')')
    ax1.xaxis.set_label_position("bottom")
    ax1.xaxis.set_label_coords(0.8, -0.18)
    ax1.titlesize: 14

    ax2.hist(df['trip size'], bins=np.arange(0,500,1), density=1) 
    ax2.set_xlim(left=0, right=500)
    ax2.set_xlabel('trip size (number of ∆ per trip)', fontsize=14)
    ax2.set_ylabel('count (number of trips)', fontsize=14)
    ax2.set_title('pdf_tripsize(' + title + ')')
    ax2.xaxis.set_label_position("bottom")
    ax2.xaxis.set_label_coords(0.8, -0.18)
    ax2.titlesize: 14
    fig.tight_layout()
    plt.savefig(os.path.join(output_path, 'trip_size (' + title + ')' + ".png"))

"""
=============
Trip duration 
=============
"""
def trip_duration(df, title, output_path):
    plt.close()
    fig = plt.figure()  # Create matplotlib figure
    ax1 = fig.add_subplot(211)  # Create matplotlib axes
    ax2 = fig.add_subplot(212)  # Create another axes that shares the same x-axis as ax.

    ax1.hist(df['duration (in min)'], cumulative=True, bins=np.arange(0, 80, 0.5), density=1) 
    ax1.set_xlabel('trip duration (in minutes)', fontsize=14)  
    ax1.set_ylabel('count (number of trips)', fontsize=14)
    ax1.set_title('cdf_trip duration (' + title + ')')
    ax1.xaxis.set_label_position("bottom")
    ax1.xaxis.set_label_coords(0.8, -0.18)
    ax1.titlesize: 14

    ax2.hist(df['duration (in min)'], bins=np.arange(0, 80, 0.5), density=1)  
    ax2.set_xlabel('trip duration (in minutes)', fontsize=14) 
    ax2.set_ylabel('count (number of trips)', fontsize=14)
    ax2.set_title('pdf_trip duration (' + title + ')')
    ax2.xaxis.set_label_position("bottom")
    ax2.xaxis.set_label_coords(0.8, -0.18)
    ax2.titlesize: 14
    fig.tight_layout()
    plt.savefig(os.path.join(output_path, 'trip_duration (' + title + ')' + ".png"))

"""
========================
Total distance traveled
in the trip 
=======================
"""
def trip_distance(df, title, output_path):
    plt.close()
    fig = plt.figure()  # Create matplotlib figure
    ax1 = fig.add_subplot(211)  # Create matplotlib axes
    ax2 = fig.add_subplot(212)  # Create another axes that shares the same x-axis as ax.

    ax1.hist(df['distance'].apply(lambda x: x/3280.8399), cumulative=True, bins=np.arange(0, 500, 1), density=1) 
    ax1.set_xlim(left=0, right=80)
    ax1.set_xlabel('trip distance (in km)', fontsize=14)  
    ax1.set_ylabel('count (number of trips)', fontsize=14)
    ax1.set_title('cdf_trip distance (' + title + ')')
    ax1.xaxis.set_label_position("bottom")
    ax1.xaxis.set_label_coords(0.8, -0.18)
    ax1.titlesize: 14

    ax2.hist(df['distance'].apply(lambda x: x/3280.8399), bins=np.arange(0, 500, 1), density=1) 
    ax2.set_xlim(left=0, right=80)
    ax2.set_xlabel('trip distance (in km)', fontsize=14) 
    ax2.set_ylabel('count (number of trips)', fontsize=14)
    ax2.set_title('pdf_trip distance (' + title + ')')
    ax2.xaxis.set_label_position("bottom")
    ax2.xaxis.set_label_coords(0.8, -0.18)
    ax2.titlesize: 14
    fig.tight_layout()
    plt.savefig(os.path.join(output_path, 'trip_distance (' + title + ')' + ".png"))

"""
=========================
Trip sample rate analysis 
=========================
"""

def trip_sample_rate(df, title, output_path):
    plt.close()

    fig = plt.figure()  # Create matplotlib figure
    ax1 = fig.add_subplot(211)  # Create matplotlib axes
    ax2 = fig.add_subplot(212)  # Create another axes that shares the same x-axis as ax.

    ax1.hist(df['sample rate'], cumulative=True, bins=np.arange(0, 20, 0.5),
             density=1)  
    ax1.set_xlabel('trip sample rate (num of hits/minute)',
                   fontsize=14)  
    ax1.set_ylabel('count (number of trips)', fontsize=14)
    ax1.set_title('cdf_trip sample rate (' + title + ')')
    ax1.xaxis.set_label_position("bottom")
    ax1.xaxis.set_label_coords(0.8, -0.18)
    ax1.titlesize: 14

    ax2.hist(df['sample rate'], bins=np.arange(0, 20, 0.5), density=1)  
    ax2.set_xlabel('trip sample rate (num of hits/minute)',
                   fontsize=14) 
    ax2.set_ylabel('count (number of trips)', fontsize=14)
    ax2.set_title('pdf_trip sample rate (' + title + ')')
    ax2.xaxis.set_label_position("bottom")
    ax2.xaxis.set_label_coords(0.8, -0.18)
    ax2.titlesize: 14
    fig.tight_layout()
    plt.savefig(os.path.join(output_path, 'trip_sample_rate (' + title + ')' + ".png"))

def create_df(files):
    """ Creates large df from multiple csv files"""
    frame = pd.DataFrame()
    df_list = []
    for file in files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except FileNotFoundError:
            print("Not including", file ,"because it does not exist")

    return pd.concat(df_list)

def size_duration_distance_sample_rate(start_date, end_date, provider, df, output_path):
    title = start_date + ' ' + end_date + ' ' + provider
    print("title: ", title)
    trip_size(df, title, output_path)
    trip_duration(df, title, output_path)
    trip_distance(df, title, output_path)
    trip_sample_rate(df, title, output_path)

def build_file(provider="", output_file="", filedate=""):
    return os.path.join(os.path.join(directory, output_folder), 
        output_file + filedate + provider + ".waynep1.csv")

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
    plt.xlabel('median trip delta', fontsize=14) 
    plt.ylabel('count (number of trips)', fontsize=14)
    plt.title(' median trip delta(' + file[145:-12] + ')')  
    plt.titlesize: 18
    plt.show()


"""
==========================
trip hourly trend analysis
Creates histogram of trip starting hours.
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
    df['DateTime'] = [convert_time(d) for d in df['start time']]
    df['Hour'] = [datetime.time(d).hour for d in df['DateTime']]
    df['Hour'].hist(cumulative=False, density=1, bins=np.arange(0, 25, 1))
    plt.xlabel('start hour', fontsize=14)
    plt.ylabel('count (number of trips)', fontsize=14)
    plt.title(' trip_hourly(' + '20171017' + ') for trip duration around 18 min')
    plt.titlesize: 18
    plt.show()

def provider_distribution(file):
    plt.close()
    df = pd.read_csv(file)
    df = df[(df['duration (in min)'] <= 15.1) & (df['duration (in min)'] >= 14.9)]
    df['provider'].value_counts().plot(kind='bar')
    plt.xlabel('provider', fontsize=14)
    plt.ylabel('count (number of trips)', fontsize=14)
    plt.title(' trip provider(' + '20171017' + ') for trip duration around 15 min')
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

if __name__ == "__main__":
    """
    python TripAnalysis.py <start_date> <end_date>
    python TripAnalysis.py 20171017 20171022

    Output will be 6 histograms in a Visualization folder in analysis_output
    """
    print('starting trip analysis')
    dates = clean_filedates(sys.argv[1:])
    start_date = dates[0]
    end_date = dates[1]

    input_folder = os.path.join('Analysis', 'data_raw')
    analysis_output = os.path.join('Analysis', 'analysis_output')
    directory = os.getcwd()

    consu_files = []
    fleet_files = []

    curr_date = start_date
    curr_year = curr_date[-8:-4]
    curr_month = curr_date[-4:-2]
    end_month = end_date[-4:-2]
    furthest_date_seen = start_date

    while curr_month <= end_month:
        month_folder = os.path.join(analysis_output, digit_month_folder(curr_date))
        output_path = os.path.join(directory, month_folder)
        for date in os.listdir(output_path):
            if date >= start_date and date <= end_date:
                output_day_folder = os.path.join(output_path, date)
                print("date added: ", date)
                consu_files.append(os.path.join(output_day_folder,
                                "trip_meta_I210." + date + "CONSUMER" + ".waynep1.csv"))
                fleet_files.append(os.path.join(output_day_folder,
                                "trip_meta_I210." + date + "FLEET" + ".waynep1.csv"))
            if date > furthest_date_seen:
                furthest_date_seen = date
        months_dict = {'1': '01', '2': '02', '3':'03', '4':'04', '5':'05', '6':'06', '7':'07',
                        '8':'08', '9':'09', '10':'10', '11':'11', '12':'12'}
        if curr_month == '12':
            next_month = '01'
            next_year = str(int(curr_year) + 1)
        else:
            next_month = months_dict[str(int(curr_month) + 1)]
            next_year = curr_year
        curr_year, curr_month = next_year, next_month
        curr_date = curr_year + curr_month + '01'
    consu_df = create_df(consu_files)
    fleet_df = create_df(fleet_files)

    visualization_folder = os.path.join(analysis_output, 'visualization')
    if not os.path.exists(os.path.join(directory, visualization_folder)):
                try:
                    os.makedirs(os.path.join(directory, visualization_folder))
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
    size_duration_distance_sample_rate(start_date, end_date, "CONSUMER", consu_df, os.path.join(directory, visualization_folder))
    size_duration_distance_sample_rate(start_date, end_date, "FLEET", fleet_df, os.path.join(directory, visualization_folder))
