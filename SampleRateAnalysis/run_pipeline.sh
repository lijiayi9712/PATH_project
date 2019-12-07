
#!/usr/bin/env bash

# Unzips and extracts data if necessary
# Runs filter or preprocess (given as first arg) on every file in data_raw subfolders
#Usage: bash run_pipeline.sh <filter/preprocess> <src>
# Example: bash month.sh filter data_raw

if [ $2 = data_raw ] 
then
	find ./data_raw/* -name "*.zip" -exec unzip -d ./data_raw/ {} \;
	py organize_data.py
	for directory in $(find ./data_raw/* -type d)
	{
		# Unzip and then delete gz files
		find . -name "*.gz" | xargs gunzip 
		# Call filter/preprocess on every date in directory
		for file in $directory/*.csv
		{ 
		date=$(echo $file | sed -r 's/.*\.(.*)\.(.*)\..*$/\1/')
		py $1.py data_raw $date
		}
	}
else
	for directory in $(find ./analysis_output/* -type d)
	{
		for date_directory in $(find $directory/* -type d)
		{	
			date=$(echo $(basename $date_directory))
			py $1.py analysis_output $date
		}
	}
fi