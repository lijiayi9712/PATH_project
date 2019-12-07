import os, sys
import errno
import shutil
import gzip
from preprocess import *

def within_bounding_box(lat, lon):
    """
    :param lat: lat
    :param lon: lon
    :return: a boolean indicating whether
    the current point is inside bounding
    box (upper left of 34.188788, -118.182502
    lower right of 34.037687, -117.855723)
    """

    if float(lat) >= 34.037687 and float(lat) <= 34.188788:
        if float(lon) >= -118.182502 and float(lon) <= -117.855723:
            return True
    return False

def is_empty_string(id, lat, lon):
    return id == '' or lat == '' or lon == ''

def is_not_empty(file):
    """ Returns true if a file has 2 or more lines """
    if not os.path.exists(file):
        return False
    lines = sum(1 for line in open(file))
    return lines > 1
    
def percentile(percent, list):
    a = np.array(list)
    return np.percentile(a, percent)

def get_heading(origin, destination):
    """
    from Serena and Alex
    Computes the heading between the origin and destination in degrees given the geolocation endpoints
    of the section as lists of lat and lon.
    Zero degrees is true north, increments clockwise.
    """
    d_lon = math.radians(destination[1])
    d_lat = math.radians(destination[0])
    o_lon = math.radians(origin[1])
    o_lat = math.radians(origin[0])

    y = math.sin(d_lon - o_lon) * math.cos(d_lat)
    x = math.cos(o_lat) * math.sin(d_lat) - \
        math.sin(o_lat) * math.cos(d_lat) * math.cos(d_lon - o_lon)
    prenormalized = math.atan2(y, x) * 180 / math.pi

    return (prenormalized + 360) % 360  # map result to [0, 360) degrees

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

def distance_traveled(speeds, trip_delta):
    """
    Computes distance = speed * time_delta
    for a list of speeds and list of trip_deltas.
    Returns the cumulative distance
    """
    total_distance = 0
    for i in range(len(trip_delta)):
        distance = speeds[i] * trip_delta[i]
        total_distance += distance
    return total_distance



""" Helpers for running on command line """


def clean_filedates(filedates):
    for i in range(len(filedates)):
        date = filedates[i]
        date = date.replace("[", "")
        date = date.replace(",", "")
        date = date.replace("]", "")
        date = date.replace(".", "")
        filedates[i] = date
    return filedates

def month_folder(filedate):
    if len(filedate) > 8:
        return filedate
    months = {'01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr', '05': 'May', '06': 'Jun',
              '07': 'July', '08': 'Aug', '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}
    year = filedate[-8:-4]
    month = months[filedate[-4:-2]]
    return month + '_' + year

def digit_month_folder(filedate):
    """ Returns the folder in format 2017_09"""
    year = filedate[-8:-4]
    month = filedate[-4:-2]
    return year + '_' + month

def create_day_folder(date):
    """ Creates the path to a day folder in analysis_output"""
    month_day = os.path.join(digit_month_folder(date), date)
    day_path = os.path.join(analysis_output, month_day)
    if not os.path.exists(os.path.join(directory, day_path)):
        try:
            os.makedirs(os.path.join(directory, day_path))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    return day_path

def build_raw_data(filedate):
    path = os.path.join(data_raw, digit_month_folder(filedate))
    return os.path.join(path, "probe_data_I210." + filedate + ".waynep1.csv")