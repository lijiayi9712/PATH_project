File Structure

Files are organized by month and by day. Each day has a folder with 11 files:
  * `bounding_box.csv` 
  * `filtered.csv` for CONSUMER and FLEET
  * `raw_trips.csv` for CONSUMER and FLEET
  * `processed.csv` for CONSUMER and FLEET
  * `trip_meta.csv` for CONSUMER and FLEET
  * `probe_meta.csv` for CONSUMER and FLEET


`bounding_box` contains one row for each probe in the raw data file with latitude and longitude within a defined I-210 Region.

`filtered` contains one row for each probe in the raw data file for points that are within the bounding box AND pass several basic data-quality checks.

`raw_trips` contains one row for each probe that was included in a trip. These trips are then filtered, producing the results in `processed`.

The `processed` file contains one row for each probe data point that met the various preprocessing parameters, with all of the information 
in the raw data file except for `system_date`.

`trip_meta` contains one row for each trip included in `processed`, ie one row for each filtered trip.

`probe_meta` contains one row for each probe included in `processed`.s