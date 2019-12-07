#!/usr/bin/env bash

# Delete all files in given month EXCEPT for bounding_box, filtered_consumer, filtered_fleet
# sample usage: bash delete_analysis.sh 2017_04

# month=$1

# find ./Analysis/analysis_output/$month -type f -name raw_trips\* -exec rm -f {} \;
# find ./Analysis/analysis_output/$month -type f -name trip_meta\* -exec rm -f {} \;
# find ./Analysis/analysis_output/$month -type f -name probe_meta\* -exec rm -f {} \;
# find ./Analysis/analysis_output/$month -type f -name processed\* -exec rm -f {} \;


#### 
# Delete entire analysis_output directory
for directory in $(find ./Analysis/analysis_output/100000*) 
{
	echo $directory
	rm -rf $directory
}