from collections import defaultdict, namedtuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dateutil import tz
import csv
import statistics
import os, errno
from decimal import Decimal



"""
bounding box: 
upper left is 34.188788, -118.182502
lower right is 34.037687, -117.855723
"""
bounding_box = lambda lat, lon: float(lat) >= 34.037687 and float(lat) <= 34.188788 and float(lon) >= -118.182502 and float(lon) <= -117.855723
"""
====PARAMETERS====
"""

minSIZE = 5  # minimum number of timestamps needed to define a trip
BARRIER = 240  # the maximum separation between consecutive timestamps within a trip (in seconds)

directory = "/Users/Lijiayi/Documents/CALPATH/SampleRateAnalysis/"

folder = 'Feb_2017_DATA/data_raw/'

output_folder = 'Feb_2017_DATA/data_raw/AnalysisOutput/'
filedate = '20170208'


"""
to analyze different
categories of data
"""
empty_filter = lambda provider: provider[0:5] == 'CONSU' or provider[0:5] == 'FLEET'
filter_func = lambda provider: provider[0:5] == 'CONSU' #and deltatime < timedelta(i) and deltatime >= timedelta(i-1)
filter_func1 = lambda provider: provider[0:5] == 'FLEET'
trip_filter = lambda tripsize, median, starttime: tripsize >= minSIZE   # and tripsize <= 50 and median >= 5

#, range_start, range_end      and starttime.hour > range_start and starttime.hour <= range_end









"""
===================================
for each probe_id, obtain the list 
of timestamps and delta time
and remove all duplicated row with
the same probe_id and sample time
===================================
"""

def timestampsbuilder(data_file, deduplicated, filter):
    Probes = namedtuple('Probes', ['provider', 'timestamps'])
    all_timestamps = {}  # probe_id (key), (provider, timestamps)
    with open(data_file, 'r') as dfile, open(deduplicated, 'w') as dfile1:
        print('Loading files [', end='', flush=True)
        reader = csv.DictReader(dfile)
        writer = csv.DictWriter(dfile1,
                                ['PROBE_ID', 'SAMPLE_DATE', 'LAT', 'LON', 'HEADING', 'SPEED', 'PROBE_DATA_PROVIDER',
                                 'SYSTEM_DATE'])
        writer.writeheader()
        progress = 0
        seen = set()
        for row in reader:
            id = row['PROBE_ID']
            lat = row['LAT']
            lon = row['LON']
            heading = row['HEADING']
            speed = row['SPEED']
            timestamp = datetime.strptime(row['SAMPLE_DATE'], '%Y-%m-%d %H:%M:%S')
            provider = row['PROBE_DATA_PROVIDER']
            systemtime = datetime.strptime(row['SYSTEM_DATE'], '%Y-%m-%d %H:%M:%S')
            if (id, timestamp) in seen:
                continue
            if systemtime < timestamp:
                continue
            elif filter(provider) and bounding_box(lat, lon):
                seen.add((id, timestamp))
                row2 = {}
                row2['PROBE_ID'] = id
                row2['SAMPLE_DATE'] = timestamp
                row2['LAT'] = lat
                row2['LON'] = lon
                row2['HEADING'] = heading
                row2['SPEED'] = speed
                writer.writerow(row2)
                if id not in all_timestamps:
                    all_timestamps[id] = Probes(provider, [timestamp])
                else:
                    all_timestamps[id].timestamps.append(timestamp)
            if progress % 10000 == 0:
                print('=', end='', flush=True)
            if progress > 10000000:
                break
            progress += 1
        print('Done!')
    return all_timestamps

"""
========================
Segment each probe's 
timestamps into trips by 
making sure the time
-delta in each trip is 
smaller than or equal to 
the predetermined maxGAP 
=========================
"""

def tripsegmentation(all_timestamps):
    AllTrips = namedtuple('AllTrips', ['provider', 'trips', 'min', 'max', 'mean', 'median'])
    Trip = namedtuple('Trip', ['probe_id', 'start_time', 'end_time', 'size', 'median'])
    all_probes = {}
    for id, Probes in all_timestamps.items():
        Probes.timestamps.sort()  # for each probe_id, sort all the timestamps
        tripSIZE = 0  # tripSIZE, tripstart are variables for each trip
        tripstart = 0
        delta_times = []  # list of delta_times which are <= maxGAP for each probe_id
        trips = []  # list of trips for each probe id
        trip_delta = []
        for i in range(1, len(Probes.timestamps), 1):  # iterate through sorted list of time stamps
            tdelta = (Probes.timestamps[i] - Probes.timestamps[
                i - 1]).total_seconds()  # take time difference (expressed in seconds) to see whether we need to restart a templist
            if tdelta <= BARRIER:
                if tripSIZE == 0:
                    tripstart = Probes.timestamps[i - 1]
                delta_times.append(tdelta)
                trip_delta.append(tdelta)
                tripSIZE += 1
            else:
                if tripSIZE >= minSIZE:  # all short trips with sizes < minSIZE would be discarded
                    trips.append(
                        Trip(probe_id=id, start_time=tripstart, end_time=Probes.timestamps[i - 1], size=tripSIZE,
                             median=trip_delta[tripSIZE // 2]))
                trip_delta.clear()
                tripSIZE = 0
        if len(trips) > 0:  # all probes without demonstrated trips are discarded
            mean = Decimal(statistics.mean(delta_times))
            mean = round(mean, 3)
            median = statistics.median(delta_times)
            if (median > 0):
                all_probes[id] = AllTrips(Probes.provider, trips, min(delta_times), max(delta_times), mean, median)
    return all_probes


"""
===================================
write the analysis results to files
===================================
"""

def writefile(outputfile, trip_meta, all_probes):
    with open(outputfile, 'w') as dfile, open(trip_meta, 'w') as dfile2:
        print('Writing', dfile.name, end='', flush=True)
        writer = csv.DictWriter(dfile, ['probe_id', 'provider', 'min', 'max', 'median_delta', 'mean_delta'])
        writer2 = csv.DictWriter(dfile2, ['trip id', 'probe id', 'start time', 'end time', 'trip size', 'duration (in min)', 'median trip delta'])
        writer.writeheader()
        writer2.writeheader()
        progress = 0
        index = 1
        for probe_id, all_trips in all_probes.items():
            row = {}
            row['probe_id'] = probe_id
            row['provider'] = all_trips.provider
            row['min'] = all_trips.min
            row['max'] = all_trips.max
            row['median_delta'] = all_trips.median
            row['mean_delta'] = round(Decimal(all_trips.mean), 3)
            writer.writerow(row)
            for trip in all_trips.trips:
                if trip_filter(trip.size, trip.median, trip.start_time):
                    row2 = {}
                    row2['trip id']     = index
                    index += 1
                    row2['probe id']     = probe_id
                    row2['start time']   = trip.start_time
                    row2['end time']     = trip.end_time
                    row2['trip size']    = trip.size
                    row2['duration (in min)']     = (trip.end_time - trip.start_time).total_seconds()/60.0
                    row2['median trip delta']    = trip.median
                    writer2.writerow(row2)
                else:
                    continue
            progress += 1
            if progress % 250 == 0:
                print('.', end='', flush=True)
        print('Done!')
    print("Writing metadata...")


"""
Main preprocess function
"""

def preprocess(minSIZE, BARRIER, filedate, type):
    if not os.path.exists(directory + output_folder):
        try:
            os.makedirs(directory + output_folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    data_file = directory + folder + "probe_data_I210." + filedate + ".waynep1.csv"    # original dataset
    deduplicated = directory + folder + "probe_data_I210_deduplicate_." + filedate + ".waynep1.csv"
    data_file1 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' + str(
        BARRIER) + ".waynep1.csv"   # + '_' + str(range1) + '_to_' + str(range2)
    data_file2 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' + str(
        BARRIER) + "CONSUMER" + ".waynep1.csv" # + '_' + str(range1) + '_to_' + str(range2)
    data_file3 = directory + output_folder + "output_probe_data_I210." + filedate + '_' + str(minSIZE) + '_' + str(
        BARRIER) + "FLEET" + ".waynep1.csv" # + '_' + str(range1) + '_to_' + str(range2)
    trip_meta = directory + output_folder + "trip_meta_I210." + filedate + '_' + str(minSIZE) + '_' + str(
        BARRIER) + ".waynep1.csv" # + '_' + str(range1) + '_to_' + str(range2)

    if type =='CONSU':
        outputfile = data_file2
        filter = filter_func
    elif type == 'FLEET':
        outputfile = data_file3
        filter = filter_func1
    elif type == 'OVERAL':
        outputfile = data_file1
        filter = empty_filter
    else:
        raise ValueError('type can only be CONSU, FLEET or OVERAL')

    all_timestamps = timestampsbuilder(data_file, deduplicated, filter)
    all_probes = tripsegmentation(all_timestamps)
    writefile(outputfile, trip_meta, all_probes)








"""
=====================
trip length analysis
=====================
"""

def triplength(file):
    plt.close()
    df = pd.read_csv(file)
    df['trip size'].describe()
    trip_size = df['trip size']
    trip_size.hist(cumulative=True, bins = 20)
    plt.xlabel('trip length (number of ∆ per trip', fontsize=14)  # 'sample rate (number of sample points/minutes)'
    plt.ylabel('count (number of trips)', fontsize=14)
    plt.title(' triplength(' + filedate + ')' + '_' + str(minSIZE) + '_' + str(
        BARRIER))
    plt.titlesize: 18
    plt.show()
    #delete the ones with unreasonably small delta time

def tripdelta(file):
    plt.close()
    df = pd.read_csv(file)
    median_delta = df['median trip delta']
    median_delta.hist(bins=20)
    plt.xlabel('median trip delta', fontsize=14)  # 'sample rate (number of sample points/minutes)'
    plt.ylabel('count (number of trips)', fontsize=14)
    plt.title(' median trip delta(' + filedate + ')' + '_' + str(minSIZE) + '_' + str(
        BARRIER))  # ' median sample rate('
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

minSIZE = 2
maxGAP = 120
preprocess(minSIZE, maxGAP, filedate, 'OVERAL')
preprocess(minSIZE, maxGAP, filedate, 'CONSU')
preprocess(minSIZE, maxGAP, filedate, 'FLEET')

minSIZE = 2
maxGAP = 240
preprocess(minSIZE, maxGAP, filedate, 'OVERAL')
preprocess(minSIZE, maxGAP, filedate, 'CONSU')
preprocess(minSIZE, maxGAP, filedate, 'FLEET')

minSIZE = 2
maxGAP = 360
preprocess(minSIZE, maxGAP, filedate, 'OVERAL')
preprocess(minSIZE, maxGAP, filedate, 'CONSU')
preprocess(minSIZE, maxGAP, filedate, 'FLEET')

minSIZE = 5
maxGAP = 120
preprocess(minSIZE, maxGAP, filedate, 'OVERAL')
preprocess(minSIZE, maxGAP, filedate, 'CONSU')
preprocess(minSIZE, maxGAP, filedate, 'FLEET')

minSIZE = 5
maxGAP = 240
preprocess(minSIZE, maxGAP, filedate, 'OVERAL')
preprocess(minSIZE, maxGAP, filedate, 'CONSU')
preprocess(minSIZE, maxGAP, filedate, 'FLEET')

minSIZE = 5
maxGAP = 360
preprocess(minSIZE, maxGAP, filedate, 'OVERAL')
preprocess(minSIZE, maxGAP, filedate, 'CONSU')
preprocess(minSIZE, maxGAP, filedate, 'FLEET')

minSIZE = 10
maxGAP = 120
preprocess(minSIZE, maxGAP, filedate, 'OVERAL')
preprocess(minSIZE, maxGAP, filedate, 'CONSU')
preprocess(minSIZE, maxGAP, filedate, 'FLEET')

minSIZE = 10
maxGAP = 240
preprocess(minSIZE, maxGAP, filedate, 'OVERAL')
preprocess(minSIZE, maxGAP, filedate, 'CONSU')
preprocess(minSIZE, maxGAP, filedate, 'FLEET')

minSIZE = 10
maxGAP = 360
preprocess(minSIZE, maxGAP, filedate, 'OVERAL')
preprocess(minSIZE, maxGAP, filedate, 'CONSU')
preprocess(minSIZE, maxGAP, filedate, 'FLEET')
"""

def transmissionTime(file):
    plt.close()
    df = pd.read_csv(file)
    df['transmissionTime'] = (pd.to_datetime(df['SYSTEM_DATE'], format='%Y-%m-%d %H:%M:%S') - pd.to_datetime(df['SAMPLE_DATE'], format='%Y-%m-%d %H:%M:%S')).dt.total_seconds()
    df['transmissionTime'].hist(bins=np.linspace(-500, 1500, 5))
    plt.xlabel('difference between system and sample time (in seconds)', fontsize=14)  # 'sample rate (number of sample points/minutes)'
    plt.ylabel('count (number of samples)', fontsize=14)
    plt.title('distribution of difference between system and sample time')
    plt.titlesize: 18
    plt.show()
    return df['transmissionTime'].describe()


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

