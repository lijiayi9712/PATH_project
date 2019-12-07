from collections import defaultdict, namedtuple
from datetime import datetime, timedelta
import csv
import pandas as pd
import statistics
import os, errno
from decimal import Decimal
import time
import math


"""
to analyze different
categories of data
"""
empty_filter = lambda provider: provider[0:5] == 'CONSU' or provider[0:5] == 'FLEET'
filter_func = lambda provider: provider[0:5] == 'CONSU'  # and deltatime < timedelta(i) and deltatime >= timedelta(i-1)
filter_func1 = lambda provider: provider[0:5] == 'FLEET'



def within_bounding_box(lat, lon):
    # upper left is 34.188788, -118.182502
    # lower right is 34.037687, -117.855723
    if float(lat) >= 34.037687 and float(lat) <= 34.188788:
        if float(lon) >= -118.182502 and float(lon) <= -117.855723:
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




def remove_NaN_values(id, lat, lon, timestamp):
    if id == '' or lat == '' or lon == '' or timestamp == '':
        return True
    return False


def filtering_and_timestamp_generation(data_file, filtered, filter):
    Probes = namedtuple('Probes', ['provider', 'timestamps', 'speeds', 'locations', 'headings'])
    all_timestamps = {}  # probe_id (key), (provider, timestamps)
    with open(data_file, 'r') as dfile, open(filtered, 'w') as dfile1:
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
            location = [lat, lon]
            heading = row['HEADING']
            speed = row['SPEED']
            timestamp = datetime.strptime(row['SAMPLE_DATE'], '%Y-%m-%d %H:%M:%S')
            provider = row['PROBE_DATA_PROVIDER']
            systemtime = datetime.strptime(row['SYSTEM_DATE'], '%Y-%m-%d %H:%M:%S')
            if (id, timestamp) in seen:
                continue
            if systemtime < timestamp:
                continue
            if remove_NaN_values(id, lat, lon, timestamp):
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


def tripsegmentation(all_timestamps, minSIZE, BARRIER, min_travel_time, min_median_speed, min_travel_distance):
    AllTrips = namedtuple('AllTrips', ['provider', 'trips', 'min', 'max', 'mean', 'median'])
    Trip = namedtuple('Trip', ['probe_id', 'start_time', 'end_time', 'size', 'median_speed', 'data', 'distance', 'provider', 'sample_rate'])
    all_probes = {}
    for id, Probes in all_timestamps.items():
        Probes.timestamps.sort()  # for each probe_id, sort all the timestamps
        tripSIZE = 0  # tripSIZE, tripstart, median_speed are variables for each trip
        tripstart = 0

        speeds = []  # list of speeds to use to calculate median_speed

        delta_times = []  # list of delta_times which are <= BARRIER for each probe_id
        trips = []  # list of trips for each probe id
        trip_delta = []

        data = []  # list that contains a list of data for each element (ie location, sample time, heading, speed)

        for i in range(1, len(Probes.timestamps), 1):  # iterate through sorted list of time stamps
            tdelta = (Probes.timestamps[i] - Probes.timestamps[i - 1]).total_seconds()
            # take time difference (expressed in seconds) to see whether we need to restart a templist
            if tdelta <= BARRIER:  # keep building trip if tdelta is small enough
                if tripSIZE == 0:
                    tripstart = Probes.timestamps[i - 1]
                # delta_times.append(tdelta)
                trip_delta.append(tdelta)
                speeds.append(float(Probes.speeds[i]))

                data.append([Probes.locations[i], Probes.timestamps[i], Probes.headings[i], Probes.speeds[i]])
                tripSIZE += 1
            else:
                if tripSIZE >= minSIZE:  # all short trips with sizes < minSIZE would be discarded
                    med_speed = float(statistics.median(speeds))
                    med_speed = round(med_speed, 3)
                    sample_rate = 60.0 / float(statistics.median(trip_delta))
                    trip = Trip(probe_id=id, start_time=tripstart, end_time=Probes.timestamps[i - 1], size=tripSIZE,
                                median_speed=med_speed, data=data, distance=distance_traveled(data), provider=Probes.provider, sample_rate=sample_rate)
                    #### Call helper functions to check valid trip
                    if valid_trip_duration(trip, min_travel_time):
                        if not vehicle_parked(trip.distance, med_speed, min_median_speed, min_travel_distance):
                            trips.append(trip)
                            delta_times = delta_times + trip_delta
                trip_delta = []
                speeds = []
                data = []
                tripSIZE = 0
        if len(trips) > 0:  # all probes without demonstrated trips are discarded
            mean = Decimal(statistics.mean(delta_times))  # mean delta time for this trip
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


def valid_trip_duration(trip, min_travel_time):
    travel_time = (trip.end_time - trip.start_time).total_seconds()
    return travel_time > min_travel_time


"""
Check that a trip has a valid median speed and travels enough distance.
Input: Trip namedtuple.
Output: True if NOT a valid trip, false otherwise
"""





def vehicle_parked(distance_traveled, med_speed, min_median_speed, min_travel_distance):
    if med_speed <= min_median_speed:
        return True
    #distance_traveled = distance_traveled(trip.data)
    if distance_traveled <= min_travel_distance:
        return True
    return False

def distance_traveled(data):
    # Use Serena and Alex's real_distance function, using the Haversine Formula

    distance_traveled = 0
    for i in range(len(data) - 2):
        loc1 = data[i][0]
        loc2 = data[i + 1][0]
        dist = real_distance(loc1, loc2)
        distance_traveled += dist  # in feet
    return distance_traveled

def real_distance(cp1, cp2):
    """
    From Serena and Alex
    Computes the distance in feet between two points using the Haversine Formula.
    :param cp1: A list in the form [lon1, lat1].
    :param cp2: A list in the form [lon2, lat2].
    :return: The distance in feet between two coordinates.
    """
    earth_radius = 6378.1
    KM_TO_FEET_CONST = 3280.84

    cp1 = list(map(math.radians, cp1))
    cp2 = list(map(math.radians, cp2))

    delta_lon = cp2[0] - cp1[0]
    delta_lat = cp2[1] - cp1[1]

    a = math.sin(delta_lat / 2) ** 2 + math.cos(cp1[1]) * math.cos(cp2[1]) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius * c * KM_TO_FEET_CONST

"""
===================================
write the analysis results to files
===================================
"""

trip_provider1 = lambda trip: trip.provider[0:5] == 'CONSUMER'
trip_provider2 = lambda trip: trip.provider[0:5] == 'FLEET'

# input_with_trip_id is the file after removing duplicates and applying bounding box

def writefile(outputfile, trip_meta, input_with_trip_id, all_probes, minSIZE, trip_provider):
    with open(outputfile, 'w') as dfile, open(trip_meta, 'w') as dfile2, open(input_with_trip_id, 'w') as dfile3:
        print('Writing files...', end='', flush=True)
        writer = csv.DictWriter(dfile, ['probe_id', 'provider', 'min', 'max', 'median_delta', 'sample_rate'])
        writer2 = csv.DictWriter(dfile2,
                                 ['trip id', 'probe id', 'start time', 'end time', 'trip size', 'duration (in min)', 'distance',
                                  'median speed', 'sample rate'])
        # writer 3 builds the input_with_trip_id, ie contains lat, lon, probe_id, trip_id
        writer3 = csv.DictWriter(dfile3, ['TRIP_ID', 'PROBE_ID', 'SAMPLE_DATE', 'LAT', 'LON', 'HEADING', 'SPEED',
                                          'PROBE_DATA_PROVIDER'])


        writer.writeheader()
        writer2.writeheader()
        writer3.writeheader()

        progress = 0
        index = 1
        trip_filter = lambda tripsize, median, starttime, provider, required_provider: tripsize >= minSIZE and (provider[0:5] == required_provider or provider[0:8] == required_provider)# and tripsize <= 50 and median >= 5

        for probe_id, all_trips in all_probes.items():
            row = {}
            row['probe_id'] = probe_id
            provider = all_trips.provider
            row['provider'] = provider
            row['min'] = all_trips.min
            row['max'] = all_trips.max
            row['median_delta'] = all_trips.median
            row['sample_rate'] = round(60.0 / float(all_trips.mean), 3)
            writer.writerow(row)
            for trip in all_trips.trips:
                if trip_filter(trip.size, trip.median_speed, trip.start_time, trip.provider, trip_provider):
                    row2 = {}
                    row2['trip id'] = index
                    row2['probe id'] = probe_id
                    row2['start time'] = trip.start_time
                    row2['end time'] = trip.end_time
                    row2['trip size'] = trip.size
                    row2['duration (in min)'] = (trip.end_time - trip.start_time).total_seconds() / 60.0
                    row2['distance'] = trip.distance
                    row2['median speed'] = trip.median_speed
                    row2['sample rate'] = trip.sample_rate
                    writer2.writerow(row2)
                    for i in range(0, trip.size, 1):
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
    return index
    # print("Writing metadata...")





"""
Main preprocess function
"""


def preprocess(minSIZE, BARRIER, min_travel_time, min_median_speed, min_travel_distance, filedate, providertype):
    start = time.time()  # Calculate elapsed time

    #directory = "/Users/Lijiayi/Documents/CALPATH/SampleRateAnalysis/"
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
            BARRIER) + '_' + str(min_travel_time) + '_' + str(min_median_speed) + '_' + str(min_travel_distance) + provider + ".waynep1.csv"

    data_file = directory + folder + "probe_data_I210." + filedate + ".waynep1.csv"  # original dataset

    filtered = directory + folder + "probe_data_I210_filtered_." + filedate + ".waynep1.csv"

    data_file1 = file_name_constructor(output_file_name="output_probe_data_I210.")
    data_file2 = file_name_constructor("CONSUMER", "output_probe_data_I210.")
    data_file3 = file_name_constructor("FLEET", "output_probe_data_I210.")
    trip_meta1 = file_name_constructor(output_file_name="trip_meta_I210.")
    trip_meta2 = file_name_constructor("CONSUMER", output_file_name="trip_meta_I210.")
    trip_meta3 = file_name_constructor("FLEET", output_file_name="trip_meta_I210.")
    input_with_trip_id1 = file_name_constructor(output_file_name="input_with_trip_id_I210.")
    input_with_trip_id2 = file_name_constructor("CONSUMER", output_file_name="input_with_trip_id_I210.")
    input_with_trip_id3 = file_name_constructor("FLEET", output_file_name="input_with_trip_id_I210.")


    if providertype == 'CONSU':
        outputfile = data_file2
        trip_meta = trip_meta2
        filter = filter_func
        input_with_trip_id = input_with_trip_id2
    elif providertype == 'FLEET':
        outputfile = data_file3
        trip_meta = trip_meta3
        filter = filter_func1
        input_with_trip_id = input_with_trip_id3
    elif providertype == 'OVERAL':
        outputfile = data_file1
        trip_meta = trip_meta1
        filter = empty_filter
        input_with_trip_id = input_with_trip_id1
    else:
        raise ValueError('type can only be CONSU, FLEET or OVERAL')

    # CALLING HELPER FUNCTIONS

    all_timestamps = filtering_and_timestamp_generation(data_file, filtered, filter)
    all_probes = tripsegmentation(all_timestamps, minSIZE, BARRIER, min_travel_time, min_median_speed, min_travel_distance)
    num_trips = writefile(outputfile, trip_meta, input_with_trip_id, all_probes, minSIZE, providertype)

    end = time.time()
    print("Elapsed Time: ", (end - start) / 60.0, "minutes")
    return num_trips


################
# COMMENT OUT THIS LINE if you don't want it to run
#preprocess(minSIZE, BARRIER, filedate, 'OVERAL')
################
