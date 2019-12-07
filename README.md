# Preprocess repository
This repository contains code to preprocess HERE data. 

## Required Programs
You will need python 3.5 and numpy installed in order to preprocess data.

You will need pandas and matplotlib to run the optional visualization functions in TripAnalysis.py

## Files in Repository
#### Preprocessing
  * preprocess.py
  * filter.py
  * utils.py
  * organize_data.py
  * organize_raw.sh
  * run_month.sh
  * run_pipeline.sh

#### Visualization and Analysis
  * TripAnalysis.py

## Outputs
Running `preprocess.py` will generate the following csv files:
  * `processed.csv` for CONSUMER and FLEET
  * `trip_meta.csv` for CONSUMER and FLEET
  * `probe_meta.csv` for CONSUMER and FLEET
  * `raw_trips.csv` for CONSUMER and FLEET
  * `bounding_box.csv` 
  * `filtered.csv` for CONSUMER and FLEET

The `processed` file contains one row for each probe data point that met the various preprocessing parameters, with all of the information in the raw data file except for `system_date`. It has one additional column `trip_id`, an integer which does not have any intrinsic value. Trip_id can be used to identify which data points belonging to a specific probe_id are in the same trip.

`trip_meta` contains one row for each trip included in `processed`, with the probe_id, provider, start time and end time of the trip, trip size (number of data points), trip duration, trip median speed, trip median speed excluding zeros, sampling rate for the trip (number of data points per minute), and trip_id. `trip_meta` can be thought of as a file containing all filtered trips. Every trip in `trip_meta` passed the trip filtering, which removed trips covering too little distance or too little time according to customizable `min_trip_distance` and `min_trip_duration` parameters.

`probe_meta` contains one row for each probe included in `processed`, with the probe_id, provider, min delta (difference in time between consecutive data points), max delta, median delta, and sampling rate for the probe.

`bounding_box` contains one row for each probe in the raw data file with latitude and longitude within a defined I-210 Region. We have the upper left corner of the box at 34.188788, -118.182502 and lower right at 34.037687, -117.855723. This includes a portion of the I-10 Freeway as well as the I-210. The row retains all columns from the raw data file except for the system date.

`filtered` contains one row for each probe in the raw data file for points that are within the bounding box AND pass several basic data-quality checks. The row retains all columns from the raw data file except for the system date.

`raw_trips` contains one row for each probe that was included in a trip. Each row contains all data included in a trip. Raw trips only need to satisfy the conditions required in defining a trip.

#### Key Terms:  
  * time delta: This refers to the difference in time (seconds) between two consecutive data points from the same probe_id  
  * trip: A sequence of consecutive data points from the same probe_id, where each pair of points is separated by no more than `maxGAP` number of seconds AND the trip includes at least `minSIZE` data points.  

## Running files

### Setup
  * Clone this repo (at minimum, download `preprocess.py`, `utils.py`, `organize_data.py`, `organize_raw.sh`)
  * Create `data_raw` and `analysis_output` directories
  * Download individual raw data files or zip folders of files from Box. Move raw data into the `data_raw` directory. Raw data files must be in the format `probe_data_I210.`date`.waynep1.csv` (or `.gz`) This should be the case if downloaded directly from Box. Files do not need to be extracted or organized into folders manually.

### Examples
#### 1. Organize
The first time you preprocess data, you can run `organize_raw.sh` in the data_raw directory. This will extract all gz files from zipped folders in the directory, move the files to the appropriate month folder (creating them if they don't already exist), expand the gz files to csv, and delete the zip folder.
```
bash organize_raw.sh
```

#### 2. Filter or Preprocess
After data has been extracted from zipped folders and is in csv format, you have more control over start and end dates.
The format is always `python <filename>.py <data source directory> <start date> <optional end date>`

Filter data from 20171017 to 20171022
```python
python filter.py data_raw 20171017 20171022
```

Preprocess data from 20171017, starting from raw data
```python
python preprocess.py data_raw 20171017
```

Preprocess data from 20171017, starting from filtered file in analysis_output
```python
python preprocess.py analysis_output 20171017
```

Preprocess data from January. 27 to Feb. 8 2017, starting from filtered data
```python
python preprocess.py analysis_output 20170127 20170208
```

#### 3. Alternatives
If you are only planning on filtering or preprocessing one month of data, you can simply use `run_month.sh`. This script will organize all raw data and then preprocess or filter the month in question starting from the `source` directory, depending on the argument you pass in.
```
bash run_month.sh <preprocess/filter> <source> 2017_07
```

`run_pipeline.sh` is similar to `run_month`, but runs either filter or preprocess on every file in the `source` directory
```
bash run_pipeline.sh <preprocess/filter> <source>
```
