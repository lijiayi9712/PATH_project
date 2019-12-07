#!/usr/bin/env bash
# 
# 1) Unzips all zip folders in data_raw
# 2) organizes data into month folders
# 3) Extracts all .gz files to csv, within the month folder.


find ./data_raw/* -name "*.zip" -exec unzip -d ./data_raw/ {} \;
rm -rf ./data_raw/*.zip
py organize_data.py
# Unzip and then delete gz files
find ./data_raw/*/* -name "*.gz" | xargs gunzip