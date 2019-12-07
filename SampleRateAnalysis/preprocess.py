from collections import namedtuple
from datetime import datetime, timedelta
import csv
import statistics
import os, errno
from decimal import Decimal
import time
import math
import numpy as np
import sys
import numpy as np
import shutil
import re
from utils import *
import atexit

""" GLOBAL VARIABLES """
directory = os.getcwd()
analysis_output = os.path.join(directory, 'analysis_output')
data_raw = os.path.join(directory, 'data_raw')


""" FILTERING """
def filtering_and_timestamp_generation(data_file, filtered, filter, bounding_box):
    """
    ===================================
    - for each probe_id, obtain list of timestamps, speeds, locations, 
    and headings ("info") and add to all_timestamps dictionary
    - remove all rows with same probe_id and sample time
    - write to bounding_box and filtered files
    ===================================

    :param data_file: source file (raw / filtered)
    :param filtered: filtered file path
    :param filter: filter for data provider
    :param bounding_box: bounding_box file path
    :return: a dictionary mapping each probe_id to its Probes in the form
    all_timestamps[id] = Probes(provider, [(timestamp1, speed1, location1, heading1),
                                           (timestamp2, speed2, location2, heading2), ...])
    """

    Probes = namedtuple('Probes', ['provider', 'info'])
    all_timestamps = {}
    progress = 0

    if data_file != filtered:
        # Set booleans to see if bounding_box file needs to be generated
        # If bounding_box already exists, use it instead of the raw data file
        if is_not_empty(bounding_box):
            input_file = bounding_box
            create_bounding_box = False
        else:
            input_file = data_file
            create_bounding_box = True
            dfile2 = open(bounding_box, 'w')
            bounding_box_writer = csv.DictWriter(dfile2, delimiter=',', lineterminator='\n',
                                          fieldnames=['PROBE_ID', 'SAMPLE_DATE', 'LAT', 'LON', 'HEADING', 'SPEED',
                                                  'PROBE_DATA_PROVIDER']) 
            bounding_box_writer.writeheader()

        with open(input_file, 'r') as dfile, open(filtered,'w') as dfile1: 
            print('Filtering file [', end='', flush=True)
            reader = csv.DictReader(dfile)
            filtered_writer = csv.DictWriter(dfile1, delimiter=',', lineterminator='\n',
                                          fieldnames=['PROBE_ID', 'SAMPLE_DATE', 'LAT', 'LON', 'HEADING', 'SPEED',
                                              'PROBE_DATA_PROVIDER']) 
            filtered_writer.writeheader()
            seen = set()
            for row in reader:
                id = row['PROBE_ID']
                lat = row['LAT']
                lon = row['LON']
                timestamp = datetime.strptime(row['SAMPLE_DATE'], '%Y-%m-%d %H:%M:%S')
                heading = row['HEADING']
                speed = row['SPEED']
                provider = row['PROBE_DATA_PROVIDER']
                location = [float(lat), float(lon)]

                def write_to_file(writer):
                    row = {}
                    row['PROBE_ID'] = id
                    row['SAMPLE_DATE'] = timestamp
                    row['LAT'] = lat
                    row['LON'] = lon
                    row['HEADING'] = heading
                    row['PROBE_DATA_PROVIDER'] = provider
                    row['SPEED'] = speed
                    writer.writerow(row)

                if not within_bounding_box(lat, lon):
                        continue
                if create_bounding_box:
                    write_to_file(bounding_box_writer)
                if (id, timestamp, lat, lon) in seen:
                    continue
                if is_empty_string(id, lat, lon):
                    continue
                if filter(provider):
                    seen.add((id, timestamp, lat, lon))
                    write_to_file(filtered_writer)
                    if id not in all_timestamps:
                        all_timestamps[id] = Probes(provider, [(timestamp, speed, location, heading)])
                    else:
                        all_timestamps[id].info.append((timestamp, speed, location, heading))

                if progress % 10000 == 0:
                    print('=', end='', flush=True)
                if progress > 10000000:
                    print("File is too large: stopping...")
                    break
                progress += 1
            print(']')
    else:
        # Use filtered file if it already exists
        with open(data_file, 'r') as dfile:
            print('Loading files [', end='', flush=True)
            reader = csv.DictReader(dfile)
            for row in reader:
                id = row['PROBE_ID']
                lat = row['LAT']
                lon = row['LON']
                heading = row['HEADING']
                speed = row['SPEED']
                timestamp = datetime.strptime(row['SAMPLE_DATE'], '%Y-%m-%d %H:%M:%S')
                provider = row['PROBE_DATA_PROVIDER']
                location = [float(lat), float(lon)]
                if id not in all_timestamps:
                    all_timestamps[id] = Probes(provider, [(timestamp, speed, location, heading)])
                else:
                    all_timestamps[id].info.append((timestamp, speed, location, heading))
                if progress % 10000 == 0:
                    print('=', end='', flush=True)
                if progress > 10000000:
                    print("File is too large: stopping...")
                    break
                progress += 1
            print(']')
    return all_timestamps


def tripsegmentation(all_timestamps, minSIZE, maxGAP, min_trip_duration, min_trip_distance, filedate,
                     provider):
    """
    ==========================================================
    Segment each probe's timestamps into trips by
    making sure the time delta in each trip is
    smaller than or equal to the predetermined maxGAP

    If a trip has number of timedelta < minSIZE, we get
    rid of the trip. Otherwise, we count it as a raw trip.

    Check whether a trip has a valid distance (> some min dist),
    and valid duration, if a trip satisfy all
    requirements, it is a qualified trip.
    ===========================================================

    :param all_timestamps: return value from initial filtering process (a dictionary mapping probe_id to probe)
    :param minSIZE: the minimum number of timestamps needed to define a trip
    :param maxGAP: the maximum separation between consecutive timestamps within each trip made by each probe id
    :param min_trip_duration: minimum number of seconds for a trip to be considered valid
    :param min_trip_distance: minimum distance traveled in feet for a trip to be considered valid
    :return: a dictionary mapping each probe_id to AllTrips(Probes.provider, trips,
                    raw_trips, min(deltas_list), max(deltas_list), mean, median)
    """

    AllTrips = namedtuple('AllTrips', ['provider', 'trips', 'raw_trips', 'min', 'max', 'mean', 'median'])
    Trip = namedtuple('Trip', ['probe_id', 'provider', 'start_time', 'end_time', 'size', 'median_speed', 'med_speed_without_zero',
                               'data', 'distance', 'sample_rate'])
    all_probes = {}
    raw_trip_count = 0
    trip_count = 0

    for id, Probes in all_timestamps.items():
        # for each probe_id, sort by timestamps
        Probes.info.sort(key=lambda x: x[0])
        tripSIZE = 0
        trip_start = 0

        trips = []  # list of trips for each probe id
        raw_trips = []  # list of all trips right after segmentation that has size >= minSIZE
        probe_deltas = []  # list of deltas which are <= maxGAP for each probe_id
        trip_deltas = [] # list of deltas in filtered trip
        raw_trip_deltas = [] # list of deltas in raw_trip
        speeds = [] # list of speeds in a trip
        data = [] # STRUCTURE: data = [[timestamp0, speed0, location0, heading0], ... ]

        # For removing points with too high distance for timedelta traveled
        earlier_delta = 0
        previous_fixed_point = None  # tuple (location, tdelta)

        # Iterate through sorted list of timestamps for one probe
        for i in range(1, len(Probes.info)):
            current_time, current_speed, current_location, current_heading = Probes.info[i][0], float(Probes.info[i][1]), Probes.info[i][2], float(Probes.info[i][3])
            previous_time, previous_speed, previous_location, previous_heading = Probes.info[i - 1][0], float(Probes.info[i-1][1]), Probes.info[i-1][2], float(Probes.info[i-1][3])
            # take time difference (expressed in seconds) to see whether we need to restart trip
            tdelta = (current_time - previous_time).total_seconds()

            if tripSIZE == 0:
                trip_start = previous_time
                previous_fixed_point = (previous_location, tdelta)

            if tdelta == 0:
                if previous_fixed_point != None and len(data) != 0:
                    distance_prev = real_distance(previous_location, previous_fixed_point[0])
                    distance_curr = real_distance(previous_fixed_point[0], current_location)
                    if distance_prev/previous_fixed_point[1] > 200 and distance_curr/previous_fixed_point[1] > 200:
                        speeds.pop(-1)
                        data.pop(-1)
                        tripSIZE -= 1
                        earlier_delta = trip_deltas.pop(-1)
                    elif distance_prev/previous_fixed_point[1] > 200 and distance_curr/previous_fixed_point[1] <= 200:
                        speeds.pop(-1)
                        data.pop(-1)
                        earlier_delta = trip_deltas.pop(-1)

                        trip_deltas.append(tdelta+earlier_delta)
                        speeds.append(current_speed)
                        data.append(
                            [current_time, current_speed, current_location, current_heading])
                continue

            if earlier_delta != 0:
                tdelta += earlier_delta
                earlier_delta = 0

            # Keep building trip if tdelta is small enough
            if tdelta <= maxGAP:
                trip_deltas.append(tdelta)
                speeds.append(current_speed)
                data.append(
                    [current_time, current_speed, current_location, current_heading])
                tripSIZE += 1
            # End trip
            if tdelta > maxGAP or (i == len(Probes.info) - 1 and tripSIZE != 0):
                if tripSIZE >= minSIZE:  # discard all short trips with sizes < minSIZE
                    med_speed = round(percentile(50, speeds), 3)
                    non_zero_speeds = [s for s in speeds if s > 0]
                    if len(non_zero_speeds) == 0:
                        med_speed_without_zero = 0
                    else:
                        med_speed_without_zero = round(percentile(50, non_zero_speeds), 3)
                    median_delta = float(statistics.median(trip_deltas))
                    sample_rate = 60.0 / median_delta
                    distance = distance_traveled(speeds, trip_deltas)
                    # if we are at the very last point for this probe, we need to force the
                    # end time to be the current probe, not the index - 1 probe.
                    if i == len(Probes.info) - 1:
                        i += 1
                    trip = Trip(probe_id=id, start_time=trip_start, end_time=Probes.info[i - 1][0],
                                size=tripSIZE, median_speed=med_speed, med_speed_without_zero=med_speed_without_zero, 
                                data=data, distance=distance, provider=Probes.provider, sample_rate=sample_rate)
                    # Always add to raw_trips
                    raw_trips.append(trip)
                    raw_trip_count += 1
                    raw_trip_deltas += trip_deltas

                    """
                    ==================
                    trip filtration
                    ==================
                    
                    """
                    # Add trip to trips if it is valid
                    if valid_trip_distance(trip.distance, min_trip_distance):
                        if valid_trip_duration(trip, min_trip_duration):
                            trips.append(trip)
                            probe_deltas += trip_deltas
                            trip_count += 1
                # Reset for new trip
                trip_deltas, speeds, data = [], [], []
                tripSIZE = 0
                previous_fixed_point = (previous_location, tdelta)
        # Add all probes that have raw_trips after for loop
        if len(raw_trips) > 0:
            if len(trips) > 0:
                deltas_list = probe_deltas
            else:
                deltas_list = raw_trip_deltas
            mean = round(Decimal(statistics.mean(deltas_list)), 3)  # mean delta time for this trip
            median = statistics.median(deltas_list)
            all_probes[id] = AllTrips(Probes.provider, trips, raw_trips, min(deltas_list), max(deltas_list), mean,
                                      median)
    return all_probes


def trip_clustering(all_probes, clusterGAP, trip_type):
    clustered_probes = {}
    AllTrips = namedtuple('AllTrips', ['provider', 'trips', 'raw_trips', 'min', 'max', 'mean', 'median'])
    Trip = namedtuple('Trip', ['probe_id', 'provider', 'start_time', 'end_time', 'size', 'median_speed',
                               'med_speed_without_zero', 'data', 'distance', 'sample_rate', 'percent_clustered'])
    for probe_id, all_trips in all_probes.items():
        clustered_trips = []
        probe_deltas = []
        if trip_type == 'filtered':
            probe_trips = all_trips.trips
        else:
            probe_trips = all_trips.raw_trips
        for trip in probe_trips:
            provider = trip.provider
            tripSIZE = 0
            trip_deltas = []
            speeds = []
            data = []
            condense_low_speed_probes = False
            previous_delta = 0
            cluster_size = 0
            for i in range(1, trip.size, 1):
                def parked(location1, location2):
                    return location1[0] == location2[0] and location1[1] == location2[1]
                def low_speed(current_speed):
                    return current_speed <= 3
                current_data = trip.data[i]
                previous_data = trip.data[i - 1]
                probe_id = probe_id
                current_stamp, current_speed, current_location, current_heading = current_data[0], current_data[1], \
                                                                                  current_data[2], current_data[3]
                previous_stamp, previous_speed, previous_location, previous_heading = previous_data[0], previous_data[1], \
                                                                                      previous_data[2], previous_data[3]
                tdelta = (current_stamp - previous_stamp).total_seconds()

                if i == 1:
                    trip_deltas.append(tdelta)
                    speeds.append(current_speed)
                    data.append([current_stamp, current_speed, current_location, current_heading])
                    tripSIZE += 1
                elif parked(current_location, previous_location) or low_speed(current_speed):
                    if i == trip.size - 1:
                        if previous_delta + tdelta <= clusterGAP:
                            # if we are the last data for this probe and need to add an incomplete cluster,
                            # we need to pass in our current delta as prev_delta + tdelta
                            trip_deltas.append(previous_delta + tdelta)
                            speeds.append(current_speed)
                            data.append([current_stamp, current_speed, current_location, current_heading])
                            cluster_size += 1
                            tripSIZE += 1
                        else:
                            # build two points: the first with prev_delta and i - 1 as the probe to take info from,
                            # the second with tdelta and i as the probe to take info from
                            trip_deltas.append(previous_delta)
                            speeds.append(previous_speed)
                            data.append([previous_stamp, previous_speed, previous_location, previous_heading])

                            trip_deltas.append(tdelta)
                            speeds.append(current_speed)
                            data.append([current_stamp, current_speed, current_location, current_heading])
                            tripSIZE += 2
                            cluster_size += 1
                    elif condense_low_speed_probes and (previous_delta + tdelta) <= maxGAP:
                        previous_delta += tdelta
                        cluster_size += 1
                    elif not condense_low_speed_probes:
                        # For edge case: first low_speed/static point after normal points
                        # add the current tdelta and starting to condense the following points
                        trip_deltas.append(tdelta)
                        speeds.append(current_speed)
                        data.append([current_stamp, current_speed, current_location, current_heading])
                        tripSIZE += 1
                        previous_delta = 0
                        condense_low_speed_probes = True
                        cluster_size += 1
                    # end a cluster
                    elif previous_delta + tdelta > maxGAP:
                        trip_deltas.append(previous_delta)
                        speeds.append(previous_speed)
                        data.append([previous_stamp, previous_speed, previous_location, previous_heading])
                        tripSIZE += 1
                        cluster_size = 1
                        previous_delta = tdelta
                else:
                    # edge case: for the first normal point after a cluster,
                    # we conclude the previous cluster before we add the current tdelta
                    if previous_delta != 0:
                        trip_deltas.append(previous_delta)
                        speeds.append(previous_speed)
                        data.append([previous_stamp, previous_speed, previous_location, previous_heading])
                        cluster_size += 1
                        tripSIZE += 1

                    # Regardless of if there was a cluster before, we need to add the current
                    # tdelta and probe i associated with it
                    trip_deltas.append(tdelta)
                    speeds.append(current_speed)
                    data.append([current_stamp, current_speed, current_location, current_heading])
                    tripSIZE += 1
                    # Reset variables
                    previous_delta = 0
                    condense_low_speed_probes = False
            # clustered trip's start time and end time stay the same as that of unclustered trip.
            # use the unclustered trip's distance while building clustered trip
            # cluster_size is the total amount of points being clustered (removed) in current unclustered trip
            # tripSIZE is the number of points in a clustered trip
            med_speed = round(percentile(50, speeds), 3)
            median_delta = float(statistics.median(trip_deltas))
            probe_deltas += trip_deltas
            non_zero_speeds = [s for s in speeds if s > 0]
            if len(non_zero_speeds) == 0:
                med_speed_without_zero = 0
            else:
                med_speed_without_zero = round(percentile(50, non_zero_speeds), 3)
            sample_rate = 60.0 / median_delta
            distance = distance_traveled(speeds, trip_deltas)
            clustered_trip = Trip(probe_id=trip.probe_id, provider=trip.provider, start_time=trip.start_time, end_time=trip.end_time,
                                  size=tripSIZE, median_speed=med_speed, med_speed_without_zero=med_speed_without_zero, data=data, distance=distance,
                                  sample_rate=sample_rate,
                                  percent_clustered=float(cluster_size / trip.size))
            clustered_trips.append(clustered_trip)
        if probe_deltas != []:
            min_delta = min(probe_deltas)
            max_delta = max(probe_deltas)
            mean_delta = round(Decimal(statistics.mean(probe_deltas)), 3)  # mean delta time for this trip
            median_delta = statistics.median(probe_deltas)
            clustered_probes[trip.probe_id] = AllTrips(provider, clustered_trips, all_trips.raw_trips, min_delta,
                                            max_delta, mean_delta, median_delta)
    return clustered_probes

""" TRIP FILTRATION HELPER FUNCTIONS """

def valid_trip_duration(trip, min_trip_duration):
    travel_time = (trip.end_time - trip.start_time).total_seconds()
    return travel_time >= min_trip_duration

def valid_trip_distance(distance_traveled, min_trip_distance):
    return distance_traveled >= min_trip_distance

def valid_trip_speed(med_speed, min_median_speed):
    return med_speed >= min_median_speed

def valid_heading(trip):
    """ Trip is valid if `percent_invalid` is less than `percentage` """
    percentage = 0.9
    count_invalid = 0
    total_count = len(trip.data)
    for i in range(1, total_count):
        previous_probe = trip.data[i - 1]
        previous_speed = previous_probe[3]
        probe = trip.data[i]
        given_heading = previous_probe[2]
        if previous_speed == 0:
            continue
        calculated_heading = get_heading(previous_probe[0], probe[0])
        if abs(float(given_heading) - calculated_heading) > 90:
            count_invalid += 1
    percent_invalid = float(count_invalid) / float(total_count)
    if percent_invalid > percentage:
        return False
    return True
"""
===================================
write the analysis results to files
===================================
"""

def writefile(probe_meta, trip_meta, processed, all_probes, trip_provider, filedate, cluster = False):
    """
    Write processed data to files

    :param outputfile: file of sample rate analysis
    :param trip_meta: file of metadata for each processed trip
    :param processed: processed data points with trip id
    :param raw_trip_file: initially filtered data points with raw trip id

    """
    write_probe_meta(probe_meta, all_probes)
    num_trips = write_trip_meta(trip_meta, all_probes, cluster)
    write_processed(processed, all_probes)
    return num_trips

def write_probe_meta(probe_meta, probes):
    with open(probe_meta, 'w') as dfile:
        probe_meta_writer = csv.DictWriter(dfile,delimiter=',', lineterminator='\n',
                                          fieldnames=['probe id', 'provider', 'min delta', 'max delta', 
                                          'median delta','sample rate'])
        probe_meta_writer.writeheader()

        for probe_id, all_trips in probes.items():
            # Write sample rate and relevant data per probe id
            provider = all_trips.provider
            row = {}
            row['probe id'] = probe_id
            row['provider'] = provider
            row['min delta'] = all_trips.min
            row['max delta'] = all_trips.max
            row['median delta'] = all_trips.median
            row['sample rate'] = round(60.0 / float(all_trips.mean), 3)
            probe_meta_writer.writerow(row)

def write_trip_meta(trip_meta, probes, cluster):
    with open(trip_meta, 'w') as dfile:
        trip_meta_writer = csv.DictWriter(dfile,delimiter=',', lineterminator='\n',
                                          fieldnames=['probe id','provider', 'start time', 'end time', 'trip size',
                                           'duration (in min)', 'distance', 'median speed','median speed without zeros', 
                                           'sample rate', 'percent clustered', 'trip id'])
        trip_meta_writer.writeheader()
        trip_id = 0
        for probe_id, all_trips in probes.items():
            for trip in all_trips.trips:
                trip_id += 1
                row2 = {}
                row2['trip id'] = trip_id
                row2['probe id'] = probe_id
                row2['provider'] =  trip.provider
                row2['start time'] = trip.start_time
                row2['end time'] = trip.end_time
                row2['trip size'] = trip.size
                row2['duration (in min)'] = (trip.end_time - trip.start_time).total_seconds() / 60.0
                row2['distance'] = trip.distance
                row2['median speed'] = trip.median_speed
                row2['median speed without zeros'] = trip.med_speed_without_zero
                row2['sample rate'] = trip.sample_rate
                if cluster:
                    row2['percent clustered'] = round(trip.percent_clustered, 3)
                else:
                    row2['percent clustered'] = 0
                trip_meta_writer.writerow(row2)
        return trip_id

def write_processed(processed, probes):
    with open(processed, 'w') as dfile:
        processed_writer = csv.DictWriter(dfile, delimiter=',', lineterminator='\n',
                                          fieldnames=['PROBE_ID', 'SAMPLE_DATE', 'LAT', 'LON',
                                                   'HEADING', 'SPEED', 'PROBE_DATA_PROVIDER','TRIP_ID'])
        processed_writer.writeheader()
        trip_id = 0
        for probe_id, all_trips in probes.items():
            for trip in all_trips.trips:
                trip_id += 1
                for i in range(0, trip.size, 1):
                    data = trip.data[i]
                    loc = data[2]
                    row3 = {}
                    row3['PROBE_ID'] = probe_id
                    row3['SAMPLE_DATE'] = data[0]
                    row3['LAT'] = loc[0]
                    row3['LON'] = loc[1]
                    row3['HEADING'] = data[3]
                    row3['SPEED'] = data[1]
                    row3['PROBE_DATA_PROVIDER'] = all_trips.provider
                    row3['TRIP_ID'] = trip_id
                    processed_writer.writerow(row3)
        return trip_id

def write_raw_file(probes, trip_provider, filedate, raw_trip_file):
    with open(raw_trip_file, 'w') as dfile4:
        raw_trips_writer = csv.DictWriter(dfile4, delimiter=',', lineterminator='\n',
                                          fieldnames=['PROBE_ID', 'SAMPLE_DATE', 'LAT',
                                                   'LON', 'HEADING', 'SPEED', 'PROBE_DATA_PROVIDER', 'TRIP_ID'])
        raw_trips_writer.writeheader()
        progress = 0
        raw_trip_id = 0
        for probe_id, all_trips in probes.items():
            # Write sample rate and relevant data per probe id
            provider = all_trips.provider
            # Write raw trips to file
            for raw_trip in all_trips.raw_trips:
                raw_trip_id += 1
                for i in range(0, raw_trip.size, 1):
                    data = raw_trip.data[i]
                    loc = data[2]
                    row4 = {}
                    row4['PROBE_ID'] = probe_id
                    row4['SAMPLE_DATE'] = data[0]
                    row4['LAT'] = loc[0]
                    row4['LON'] = loc[1]
                    row4['HEADING'] = data[3]
                    row4['SPEED'] = data[1]
                    row4['PROBE_DATA_PROVIDER'] = provider
                    row4['TRIP_ID'] = raw_trip_id
                    raw_trips_writer.writerow(row4)
        return raw_trip_id

"""
Main preprocess function
"""
def preprocess(minSIZE, maxGAP, min_trip_duration, min_trip_distance, filedate, providertype,
               cluster=False, cluster_from='filtered'):
    output_folder = create_day_folder(filedate)

    def build_file(provider="", output_file=""):
        return os.path.join(output_folder, output_file + filedate + provider + ".waynep1.csv")

    bounding_box = build_file(output_file='bounding_box_.')

    filtered1 = build_file(output_file="filtered_.")
    filtered2 = build_file("CONSUMER", "filtered_.")
    filtered3 = build_file("FLEET", "filtered_.")

    raw_trips1 = build_file(output_file="raw_trips_I210.")
    raw_trips2 = build_file("CONSUMER", output_file="raw_trips_I210.")
    raw_trips3 = build_file("FLEET", output_file="raw_trips_I210.")

    probe_meta1 = build_file(output_file="probe_meta_I210.")
    probe_meta2 = build_file("CONSUMER", "probe_meta_I210.")
    probe_meta3 = build_file("FLEET", "probe_meta_I210.")

    trip_meta1 = build_file(output_file="trip_meta_I210.")
    trip_meta2 = build_file("CONSUMER", output_file="trip_meta_I210.")
    trip_meta3 = build_file("FLEET", output_file="trip_meta_I210.")

    processed1 = build_file(output_file="processed_I210.")
    processed2 = build_file("CONSUMER", output_file="processed_I210.")
    processed3 = build_file("FLEET", output_file="processed_I210.")

    clustered1 = build_file(output_file="processed_clustered_I210.")
    clustered2 = build_file("CONSUMER", output_file="processed_clustered_I210.")
    clustered3 = build_file("FLEET", output_file="processed_clustered_I210.")
    clustered_trip_meta1 = build_file(output_file="clustered_trip_meta_I210.")
    clustered_trip_meta2 = build_file("CONSUMER", output_file="clustered_trip_meta_I210.")
    clustered_trip_meta3 = build_file("FLEET", output_file="clustered_trip_meta_I210.")
    clustered_probe_meta1 = build_file(output_file="clustered_probe_meta_I210.")
    clustered_probe_meta2 = build_file("CONSUMER", "clustered_probe_meta_I210.")
    clustered_probe_meta3 = build_file("FLEET", "clustered_probe_meta_I210.")

    if providertype == 'OVERAL':
        provider_filter = lambda provider: provider[0:5] == 'CONSU' or provider[0:5] == 'FLEET'
        filtered = filtered1
        probe_meta = probe_meta1
        raw_trip_file = raw_trips1
        trip_meta = trip_meta1
        processed = processed1
        clustered = clustered1
        clustered_probe_meta = clustered_probe_meta1
        clustered_trip_meta = clustered_trip_meta1

    elif providertype == 'CONSU':
        provider_filter = lambda provider: provider[0:5] == 'CONSU'
        filtered = filtered2
        probe_meta = probe_meta2
        raw_trip_file = raw_trips2
        trip_meta = trip_meta2
        processed = processed2
        clustered = clustered2
        clustered_probe_meta = clustered_probe_meta2
        clustered_trip_meta = clustered_trip_meta2

    elif providertype == 'FLEET':
        provider_filter = lambda provider: provider[0:5] == 'FLEET'
        filtered = filtered3
        probe_meta = probe_meta3
        raw_trip_file = raw_trips3
        trip_meta = trip_meta3
        processed = processed3
        clustered = clustered3
        clustered_probe_meta = clustered_probe_meta3
        clustered_trip_meta = clustered_trip_meta3

    else:
        raise ValueError('type can only be CONSU, FLEET or OVERAL')

    # If filtered file was already made, use it for further processing
    if is_not_empty(filtered):
        starting_file = filtered
    else:
        starting_file = build_raw_data(filedate)

    # FILTER box
    all_timestamps = filtering_and_timestamp_generation(starting_file, filtered, provider_filter, bounding_box)

    # TRIP SEGMENTATION AND TRIP FILTERING
    all_probes = tripsegmentation(all_timestamps, minSIZE, maxGAP, min_trip_duration,
                                  min_trip_distance, filedate, providertype)

    num_raw_trips = write_raw_file(all_probes, providertype, filedate, raw_trip_file)

    # OPTIONAL CLUSTERING
    if cluster:
        clustered_probes = trip_clustering(all_probes, maxGAP, cluster_from)
        num_trips = writefile(clustered_probe_meta, clustered_trip_meta, clustered, clustered_probes,
                              providertype, filedate, cluster)
    # WRITE TO FILES
    else:
        num_trips = writefile(probe_meta, trip_meta, processed, all_probes,
                                                 providertype, filedate)

    print('There were', num_raw_trips, ' raw trips(' + str(providertype) + ') on ' + str(filedate) + '.')
    print('There were', num_trips, ' trips(' + str(providertype) + ') on ' + str(filedate) + '.')
    return (num_trips, num_raw_trips)

# def exit_handler():
#     print('if the current file is partially done, please delete all its generated files '
#           'and rerun the date in progress to prevent preprocessing errors.')

# atexit.register(exit_handler)

if __name__ == "__main__":
    """
    >>> python preprocess.py data_raw 20171001 20171022
    Runs preprocessing from <start_date> to <end_date>, with data starting from data_raw or analysis_output
    If only <start_date> is provided, then only runs preprocessing on that date.

    """
    minSIZE = 5
    maxGAP = 240
    min_trip_distance = 1000 # minimum number of feet for a trip to be considered valid
    min_trip_duration = 180  # minimum number of seconds for a trip to be considered valid

    data_start = sys.argv[1]
    data_location = os.path.join(directory, data_start)
    filedates = clean_filedates(sys.argv[2:])
    start_date = filedates[0]
    if len(sys.argv) > 3:
        end_date = filedates[1]
        print("Preprocessing data from {0} to {1}".format(start_date, end_date))
    else:
        end_date = start_date
        print("Preprocessing data from {0}".format(start_date))
    curr_date = start_date
    curr_year, curr_month  = curr_date[-8:-4], curr_date[-4:-2]
    end_year, end_month = end_date[-8:-4], end_date[-4:-2]
    furthest_date_seen = start_date

    while curr_month <= end_month or curr_year < end_year:
        month_folder =  digit_month_folder(curr_date)
        if not os.path.exists(os.path.join(analysis_output, month_folder)):
            try:
                os.makedirs(os.path.join(analysis_output, month_folder))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        data_input_path = os.path.join(data_location, month_folder)
        for date in os.listdir(data_input_path):
            # if data_input_path is analysis_output, the 'date' will be the name of the day folder.
            # Otherwise, need to get date out of probe_data_I210.<date>.waynep1.csv format
            if data_start == 'data_raw':
                date = date[-20:-12]
            if date >= start_date and date <= end_date:
                try: 
                    print("\nRunning preprocess on", date)


                    preprocess(minSIZE, maxGAP, min_trip_duration, min_trip_distance, date, 'CONSU', cluster=False)
                    preprocess(minSIZE, maxGAP, min_trip_duration, min_trip_distance, date, 'FLEET', cluster=False)
                except FileNotFoundError:
                    print("Unable to preprocess {0}: data from {0} was not found.".format(date))
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
