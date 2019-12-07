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
import time
import math

"""
====PARAMETERS====
"""

minSIZE = 5  # minimum number of timestamps needed to define a trip
BARRIER = 240  # the maximum separation between consecutive timestamps within a trip (in seconds)
min_travel_time = 180 # minimum number of seconds for a trip to be considered valid
min_median_speed = 5.0 # minimum median speed for a trip to be considered valid
min_travel_distance = 1000 # travel distance in feet, 1000ft to 0.5mi (2 or 3 blocks, Qijian asks) 
# (test sensitivity for min_travel_dist)

### CHANGE THESE FOR YOUR SETUP
directory = "/Users/amymendelsohn/su18/cc/SampleAnalysis/"

folder = 'Feb_2017_DATA/data_raw/'

output_folder = 'Feb_2017_DATA/data_raw/AnalysisOutput/'
filedate = '20170208'

### RUN THIS COMMAND TO PREPROCESS
#preprocess(minSIZE, BARRIER, filedate, 'OVERAL')


"""
to analyze different
categories of data
"""
empty_filter = lambda provider: provider[0:5] == 'CONSU' or provider[0:5] == 'FLEET'
filter_func = lambda provider: provider[0:5] == 'CONSU' #and deltatime < timedelta(i) and deltatime >= timedelta(i-1)
filter_func1 = lambda provider: provider[0:5] == 'FLEET'
trip_filter = lambda tripsize, median, starttime: tripsize >= minSIZE   # and tripsize <= 50 and median >= 5

def within_bounding_box(lat, lon):
    # upper left is 34.188788, -118.182502
    # lower right is 34.037687, -117.855723
    if float(lat) > 34.037687 and float(lat) < 34.188788:
        if float(lon) > -118.182502 and float(lon) < -117.855723:
            return True
    return False
"""
===================================
- for each probe_id, obtain the list 
of timestamps and delta time
- remove all duplicated rows with
the same probe_id and sample time
- add bounding box
- save to a new file
===================================
"""

def filtering_and_timestamp_generation(data_file, deduplicated, filter):
    Probes = namedtuple('Probes', ['provider', 'timestamps', 'speeds', 'locations', 'headings'])
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
            if not within_bounding_box(lat, lon):
                continue
            location = (lat, lon)
            heading = row['HEADING']
            speed = row['SPEED']
            timestamp = datetime.strptime(row['SAMPLE_DATE'], '%Y-%m-%d %H:%M:%S')
            provider = row['PROBE_DATA_PROVIDER']
            systemtime = datetime.strptime(row['SYSTEM_DATE'], '%Y-%m-%d %H:%M:%S')
            if (id, timestamp) in seen:
                continue
            if systemtime < timestamp:
                continue
            elif filter(provider):
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
                    all_timestamps[id] = Probes(provider, [timestamp], [speed], [location], [heading])
                else:
                    all_timestamps[id].timestamps.append(timestamp)
                    all_timestamps[id].speeds.append(speed)
                    all_timestamps[id].locations.append(location)
                    all_timestamps[id].headings.append(heading)
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
the predetermined BARRIER

Also, check that trip is a 
valid distance (> some min dist)
and valid duration
=========================
"""

def tripsegmentation(all_timestamps):
    AllTrips = namedtuple('AllTrips', ['provider', 'trips', 'min', 'max', 'mean', 'median'])
    Trip = namedtuple('Trip', ['probe_id', 'start_time', 'end_time', 'size', 'median_speed', 'data'])
    all_probes = {}
    for id, Probes in all_timestamps.items():
        Probes.timestamps.sort()  # for each probe_id, sort all the timestamps
        tripSIZE = 0  # tripSIZE, tripstart, median_speed are variables for each trip
        tripstart = 0

        med_speed = 0.0
        speeds = [] # list of speeds to use to calculate median_speed

        delta_times = []  # list of delta_times which are <= BARRIER for each probe_id
        trips = []  # list of trips for each probe id
        trip_delta = []

        data = [] # list that contains a list of data for each element (ie location, sample time, heading, speed)

        for i in range(1, len(Probes.timestamps), 1):  # iterate through sorted list of time stamps
            tdelta = (Probes.timestamps[i] - Probes.timestamps[
                i - 1]).total_seconds()  # take time difference (expressed in seconds) to see whether we need to restart a templist
            if tdelta <= BARRIER: #keep building trip if tdelta is small enough
                if tripSIZE == 0:
                    tripstart = Probes.timestamps[i - 1]
                delta_times.append(tdelta)
                trip_delta.append(tdelta)
                speeds.append(Probes.speeds[i])
                data.append([Probes.locations[i], Probes.timestamps[i], Probes.headings[i], Probes.speeds[i]])
                tripSIZE += 1
            else:
                if tripSIZE >= minSIZE:  # all short trips with sizes < minSIZE would be discarded
                    med_speed = float(speeds[tripSIZE // 2])
                    trip = Trip(probe_id=id, start_time=tripstart, end_time=Probes.timestamps[i - 1], size=tripSIZE,
                                 median_speed=med_speed, data=data)
                    #### Call helper functions to check valid trip
                    if valid_trip_duration(trip):
                        if not vehicle_parked(trip, med_speed):
                            trips.append(trip)
                trip_delta.clear()
                tripSIZE = 0
        if len(trips) > 0:  # all probes without demonstrated trips are discarded
            mean = Decimal(statistics.mean(delta_times)) # mean delta time for this trip
            mean = round(mean, 3)
            median = statistics.median(delta_times)
            if (median > 0):
                all_probes[id] = AllTrips(Probes.provider, trips, min(delta_times), max(delta_times), mean, median)
    return all_probes

"""
Check that a trip travels for enough TIME to be considered a valid trip.
Note that min_travel_time is in seconds
Input: Trip namedtuple.
Output: True if valid trip, False otherwise
"""
def valid_trip_duration(trip):
    travel_time = (trip.end_time - trip.start_time).total_seconds()
    return travel_time > min_travel_time

"""
Check that a trip has a valid median speed and travels enough distance.
Input: Trip namedtuple.
Output: True if NOT a valid trip, false otherwise
"""
def vehicle_parked(trip, med_speed):
    if med_speed < min_median_speed:
        return True
    # Compute distance: http://jonisalonen.com/2014/computing-distance-between-coordinates-can-be-simple-and-fast/
    # This is currently using an APPROXIMATION
    # Non-approximation python: https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
    degree_len = 110.25
    start_time = time.time()

    distance_traveled = 0
    for i in range(len(trip.data) - 2):
        loc1 = trip.data[i][0]
        loc2 = trip.data[i + 1][0]
        x = float(loc1[0]) - float(loc2[0])
        y = (float(loc1[1]) - float(loc2[1])) * math.cos(float(loc1[0]))
        km_to_feet = 3280.84
        distance_traveled += degree_len * math.sqrt(x*x + y*y) * km_to_feet # in feet

    if distance_traveled < min_travel_distance:
        return True
    return False



"""
===================================
write the analysis results to files
===================================
"""

def writefile(outputfile, trip_meta, input_with_trip_id, all_probes, deduplicated):
    with open(outputfile, 'w') as dfile, open(trip_meta, 'w') as dfile2, open(input_with_trip_id, 'w') as dfile3:
        print('Writing files...', end='', flush=True)
        writer = csv.DictWriter(dfile, ['probe_id', 'provider', 'min', 'max', 'median_delta', 'mean_delta'])
        writer2 = csv.DictWriter(dfile2, ['trip id', 'probe id', 'start time', 'end time', 'trip size', 'duration (in min)', 'median speed'])
        # writer 3 builds the input_with_trip_id, ie contains lat, lon, probe_id, trip_id
        writer3 = csv.DictWriter(dfile3, ['TRIP_ID', 'PROBE_ID', 'SAMPLE_DATE', 'LAT', 'LON', 'HEADING', 'SPEED', 'PROBE_DATA_PROVIDER'])

        # use deduplicated as the basis for building input_with_trip_id : just add the trip_id!
        # reader = csv.DictReader(deduplicated)
        # Uncomment this later if you want to continue implementing this path

        writer.writeheader()
        writer2.writeheader()
        writer3.writeheader()

        progress = 0
        index = 1
        for probe_id, all_trips in all_probes.items():
            row = {}
            row['probe_id'] = probe_id
            provider = all_trips.provider
            row['provider'] = provider
            row['min'] = all_trips.min
            row['max'] = all_trips.max
            row['median_delta'] = all_trips.median
            row['mean_delta'] = round(Decimal(all_trips.mean), 3)
            writer.writerow(row)
            for trip in all_trips.trips:
                if trip_filter(trip.size, trip.median_speed, trip.start_time):
                    row2 = {}
                    row2['trip id']     = index
                    row2['probe id']     = probe_id
                    row2['start time']   = trip.start_time
                    row2['end time']     = trip.end_time
                    row2['trip size']    = trip.size
                    row2['duration (in min)']     = (trip.end_time - trip.start_time).total_seconds()/60.0
                    row2['median speed']    = trip.median_speed
                    writer2.writerow(row2)
                    for i in range(trip.size):
                        data = trip.data[i]
                        loc = data[0]
                        row3 = {}
                        row3['TRIP_ID'] = index
                        row3['PROBE_ID'] = probe_id
                        row3['SAMPLE_DATE'] = data[1]
                        row3['LAT'] = loc[0]
                        row3['LON'] = loc[1]
                        row3['HEADING'] = data[2]
                        row3['SPEED'] = data[3]
                        row3['PROBE_DATA_PROVIDER'] = provider
                        writer3.writerow(row3)
                    index += 1

                else:
                    continue
            progress += 1
            if progress % 250 == 0:
                print('.', end='\n', flush=True)
        print('Done! There were', index, 'trips.')
    #print("Writing metadata...")


"""
Main preprocess function
"""

def preprocess(minSIZE, BARRIER, filedate, type):
    
    start = time.time() # Calculate elapsed time

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
    input_with_trip_id = directory + output_folder + "input_with_trip_id_I210." + filedate + '_' + str(minSIZE) + '_' + str(
        BARRIER) + ".waynep1.csv"

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


    # CALLING HELPER FUNCTIONS

    all_timestamps = filtering_and_timestamp_generation(data_file, deduplicated, filter)
    all_probes = tripsegmentation(all_timestamps)
    writefile(outputfile, trip_meta, input_with_trip_id, all_probes, deduplicated)

    end = time.time()
    print("Elapsed Time: ", (end - start)/60.0, "minutes")

################
# COMMENT OUT THIS LINE if you don't want it to run
preprocess(minSIZE, BARRIER, filedate, 'OVERAL')
################



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
BARRIER = 120
preprocess(minSIZE, BARRIER, filedate, 'OVERAL')
preprocess(minSIZE, BARRIER, filedate, 'CONSU')
preprocess(minSIZE, BARRIER, filedate, 'FLEET')
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

