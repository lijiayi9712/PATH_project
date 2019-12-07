from collections import namedtuple
from datetime import datetime
import csv
import os, sys, errno, time
from utils import *
import atexit

""" GLOBAL VARIABLES """
directory = os.getcwd()
analysis_output = os.path.join(directory, 'analysis_output')
data_raw = os.path.join(directory, 'data_raw')

def create_filtered(data_file, filtered, filter, bounding_box):
    """
    Creates the bounding_box, filtered_consumer, filtered_fleet csv files.
    Does not create or return the all_timestamps dictionary mapping probe_id to probe data. 

    :param data_file: source file (raw / filtered)
    :param filtered: filtered file path
    :param filter: filter for data provider
    :param bounding_box: bounding_box file path
    """
    progress = 0

    # If filtered file already exists, we do not create either a bounding_box or filtered file
    if is_not_empty(filtered) and is_not_empty(bounding_box):
        print("Filtered file already exists: moving on...")
        return
    # Set booleans to see if bounding_box file needs to be generated along the way.
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

    with open(input_file, 'r') as dfile, open(filtered, 'w') as dfile1:
        print('Filtering [', end='', flush=True)
        reader = csv.DictReader(dfile)
        filtered_writer = csv.DictWriter(dfile1, delimiter=',', lineterminator='\n',
                                          fieldnames= ['PROBE_ID', 'SAMPLE_DATE', 'LAT', 'LON', 'HEADING', 'SPEED',
                                          'PROBE_DATA_PROVIDER'])
        filtered_writer.writeheader()
        seen = set()

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

        for row in reader:
            id = row['PROBE_ID']
            lat = row['LAT']
            lon = row['LON']
            timestamp = datetime.strptime(row['SAMPLE_DATE'], '%Y-%m-%d %H:%M:%S')
            heading = row['HEADING']
            speed = row['SPEED']
            provider = row['PROBE_DATA_PROVIDER']
            if create_bounding_box:
                if not within_bounding_box(lat, lon):
                    continue
                write_to_file(bounding_box_writer)

            if (id, timestamp) in seen or is_empty_string(id, lat, lon):
                continue
            if filter(provider):
                seen.add((id, timestamp))
                write_to_file(filtered_writer)

            if progress % 10000 == 0:
                print('=', end='', flush=True)
            if progress > 10000000:
                break
            progress += 1
        print('Done!')


# def exit_handler():
#     print('if the current file is partially done, please rerun the date in progress to prevent preprocessing errors.')

# atexit.register(exit_handler)

if __name__ == "__main__":
    """
    >>> python filter.py data_raw 20171001 20171022
    Runs preprocessing from <start_date> to <end_date>, using data in the data_raw directory.
    If only <start_date> is provided, then only runs filtering on that date.

    """
    minSIZE = 5
    maxGAP = 240
    min_trip_distance = 1000 # minimum number of feet for a trip to be considered valid
    min_trip_duration = 180  # minimum number of seconds for a trip to be considered valid

    filedates = clean_filedates(sys.argv[2:])
    start_date = filedates[0]
    if len(sys.argv) > 3:
        end_date = filedates[1]
        print("\nFiltering data from {0} to {1}".format(start_date, end_date))
    else:
        end_date = start_date
        print("\nFiltering data from {0}".format(start_date))

    curr_date = start_date
    curr_year, curr_month  = curr_date[-8:-4], curr_date[-4:-2]
    end_year, end_month = end_date[-8:-4], end_date[-4:-2]
    furthest_date_seen = start_date

    while curr_month <= end_month or curr_year < end_year:
        month_folder = digit_month_folder(curr_date)
        output_path = os.path.join(analysis_output, month_folder)
        data_input_path = os.path.join(data_raw, month_folder)
        for date in os.listdir(data_input_path):
            date = date[-20:-12]
            if date >= start_date and date <= end_date:
                try: 
                    raw_data = build_raw_data(date)
                    for provider in ['CONSUMER', 'FLEET']:
                        print("Running filtering on date:", date, provider)
                        day_path = create_day_folder(date)
                        bounding_box = os.path.join(os.path.join(directory, day_path), "bounding_box_." + date + ".waynep1.csv")
                        filtered = os.path.join(os.path.join(directory, day_path), "filtered_." + date + provider + ".waynep1.csv")
                        if provider[0:5] == 'CONSU':
                            provider_filter = lambda provider: provider[0:5] == 'CONSU'
                        else:
                            provider_filter = lambda provider: provider[0:5] == 'FLEET'
                        create_filtered(raw_data, filtered, provider_filter, bounding_box)
                except FileNotFoundError:
                    print("Unable to filter {0}: data from {0} was not found.".format(date))
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
