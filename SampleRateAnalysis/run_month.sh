#!/usr/bin/env bash
# 
# 1) Unzips and extracts all data in data_raw, if any data is unorganized
# 2) Runs filter or preprocess (given as first arg) on every date in the given month
# 
# Usage: bash month.sh <filter/preprocess> <src> <start month> (optional: more months)
# Example: bash month.sh filter data_raw 2017_08


if [ $2 = data_raw ] 
then
	find ./data_raw/* -name "*.zip" -exec unzip -d ./data_raw/ {} \;
	py organize_data.py
	# Unzip and then delete gz files
	find ./data_raw/*/* -name "*.gz" | xargs gunzip 
	for month in "${@:3}" 
	{
		echo "month:"
		echo $month
		files=$(echo $(find ./data_raw/$month/*))
		for file in $(find ./data_raw/$month/*)
		{ 
			date=$(echo $file | sed -r 's/.*\.(.*)\.(.*)\..*$/\1/')
			py $1.py data_raw $date
		}
	}
else
	for month in "${@:3}" 
	{
		for date_directory in $(find ./analysis_output/$month/* -type d)
		{	
			date=$(echo $(basename $date_directory))
			echo "date:"
			echo $date
			py $1.py analysis_output $date
		}
	}