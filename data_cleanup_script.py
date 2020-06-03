"""
# Data Cleaning for the Full Summary File of the Million Song Dataset

The goal of this script is to perform the data cleaning for our project using the Million Song Dataset (MSD). Given that the size of the full dataset is around 280 GB, we are using the provided summary file which is approximately 300 MB and contains the metadata for all the songs, minus some of the audio analysis and other features which are not relevant for our project.
"""

"""
## Cleanup Step 1: Only include songs with lyrics
We only want to consider the tracks that we also have the bag-of-words lyrics for. First we need to parse the bag-of-words file to extract all the Track IDs, and then we can find the metadata for those tracks.
"""

import glob
import os
from pprint import pprint # for nicely printed lists
import random # for printing examples


def getTrackIDsFromBagOfWordsFile(filepath):
    print('Retrieving Track IDs from ', filepath)
    track_id_list = []
    with open(filepath, 'r') as f:
        for line in f:
        # reading the file line-by-line
            if line[0] == "#" or line[0] == "%":
                continue # ignore lines that are comments or does not contain a track id
            else:
                # this lines starts with the track id, followed by a commma
                comma_idx = line.find(',') # returns index of 1st occurence in string
                track_id = line[:comma_idx]
                track_id_list.append(track_id)
    return track_id_list

train_bag_of_words_filepath = './mxm_dataset_train.txt'
test_bag_of_words_filepath = './mxm_dataset_test.txt'

track_ids_train = getTrackIDsFromBagOfWordsFile(train_bag_of_words_filepath)
track_ids_test = getTrackIDsFromBagOfWordsFile(test_bag_of_words_filepath)

lyric_track_ids = track_ids_train + track_ids_test
print("\nNumber of track IDs with lyrics: ", len(lyric_track_ids))

print("\nNumber of track IDs in original training set: ", len(track_ids_train))
print("Example track IDs found: ")
for i in range(5):
    pprint(random.choice(track_ids_train))

print("\nNumber of track IDs in original testing set: ", len(track_ids_test))
print("Example track IDs found: ")
for i in range(5):
    pprint(random.choice(track_ids_test))

"""In total, it seems like we have bag-of-words lyrics for 237,662 tracks, for the entire dataset. We need to find the metadata corresponding to these tracks.

The summary file is `.h5` format, so it will need to be converted into a common format such as `numpy` array or `pandas` dataframe which will be more compatible with other common libraries. Info about the fields provided within the file can be found here: http://millionsongdataset.com/pages/field-list/, but may not apply to the summary file.
<br>
<br>
The dataset recommends using a wrapper file to easily access the fields in the h5 dataset, so that is what is being imported and used below. Note that this requires you to have pytables, hdf5, numpy, numexpr, and cython installed in your environment. Additionally, the `hdf5_getters.py` may need to be modified to use `tables.open_file` instead of `tables.openFile`
<br> <br>
More info can be found here:
<br>http://millionsongdataset.com/pages/code/#wrappers <br>http://www.pytables.org/usersguide/installation.html

## The code in the cell below takes a very long time to run (multiple hours on my machine) so don't run it if it isn't absolutely necessary for you to re-extract all the metadata!
"""

import hdf5_getters # provided by the Million Song Dataset website
import numpy as np
from tqdm import tqdm

filepath = './msd_summary_file.h5'

num_songs = 1000000 # one millions songs in the dataset
num_threads = 10 # multithreading will speed up the process
tracks_per_thread = int(num_songs / num_threads)

arr = np.empty((0,8)) # shape (0, N), start with 0 rows with N features per row
lyric_track_ids_set = set(lyric_track_ids)

# for iteration in progress_bar:
def getTrackInfo(starting_num):
    my_list = []
    f = hdf5_getters.open_h5_file_read(filepath)
    progress_bar = tqdm(range(tracks_per_thread))
    for iteration in progress_bar:
        i = int(iteration) + (starting_num*tracks_per_thread)
        track_id = hdf5_getters.get_track_id(f, i).decode()
        if track_id not in lyric_track_ids_set:
            continue # skip it an go on
        artist_name = hdf5_getters.get_artist_name(f, i).decode()
        duration = hdf5_getters.get_duration(f, i)
        loudness = hdf5_getters.get_loudness(f, i)
        tempo = hdf5_getters.get_tempo(f, i)
        title = hdf5_getters.get_title(f, i).decode()
        year = hdf5_getters.get_year(f, i)
        long_list = [track_id, artist_name, duration, loudness, tempo, title, year]
        my_list.append(long_list)
        progress_bar.set_description("Iteration %d" % i)
    f.close()
    return my_list

print("Starting data extraction with multiprocessing")
import multiprocessing
output = []
with multiprocessing.Pool(16) as pool:
    output = pool.map(getTrackInfo, range(num_threads))

print("Combining results into single list")
output_list = [long_list for thread_list in output for long_list in thread_list]
arr = np.array(output_list).reshape(-1, 7) # we extract 7 features of the track in getTrackInfo()

print("Final Array Shape: ", arr.shape)
print("Example array entries: ")

for i in range(10):
    pprint(random.choice(arr))

import pandas as pd

df = pd.DataFrame(arr, columns=["track_id", "artist_name", "duration", "loudness", "tempo", "title", "year"])

print(df.shape)
print(df.columns)
print(df.head())


"""In the next cell, we will extract the bag-of-words for each of the track ids."""

def getBagOfWordsFromFile(filepath):
    # filepath is the path to the bag-of-words file

    track_bag_dict = dict() # key = track ID, value = bag of words
    with open(filepath, 'r') as f:
        for line in f:
            if line[0] != '#' and line[0] != '%':
                comma_idx = line.find(',')
                track_id = line[:comma_idx]
                second_comma_idx = line.find(',', comma_idx+1) # find comma before the start of the bag of words
                bag_of_words = line[second_comma_idx+1:-1] # -1 removes the newline \n at the end of the line
                track_bag_dict[track_id] = bag_of_words
    return track_bag_dict

test_bag_of_words_filepath = './mxm_dataset_test.txt'
train_bag_of_words_filepath = './mxm_dataset_train.txt'

print("Getting bag-of-words from test set...")
test_bag_dict = getBagOfWordsFromFile(test_bag_of_words_filepath)
print("Getting bag-of-words from training set...")
train_bag_dict = getBagOfWordsFromFile(train_bag_of_words_filepath)
print("Finished parsing files.")

# combining the two dicts together
track_bag_of_words_dict = {**train_bag_dict, **test_bag_dict}

print("# Tracks found with lyrics: ", len(track_bag_of_words_dict))

"""
Now we need to add the bag of words to the dataframe with the metadata.
"""

# giving an order to the dict items
track_ids = df['track_id'].tolist()
converted_array = [track_bag_of_words_dict[track_id] for track_id in track_ids]

# add it to the dataframe
df['lyrics'] = converted_array

"""
Now we need to add the genre tags to the dataframe with the remaining metadata.
"""

def extractFiles(path):
    filepathList = []
    for filename in glob.iglob(path, recursive=True):
        if os.path.isfile(filename): # filters out directories
            filepathList.append(filename)
    return filepathList

# File path to the data folder containing the genre information
# Change as needed
path_train_json = './lastfm_train/**'
path_test_json = './lastfm_test/**'
json_filepaths_train = extractFiles(path_train_json)
json_filepaths_test = extractFiles(path_test_json)

json_filepaths = json_filepaths_train + json_filepaths_test

print("Num JSON Files: ", len(json_filepaths))
print("Example JSON file paths: ")

for i in range(10):
    pprint(random.choice(json_filepaths))


import simplejson
# make chunks to use for multithreading
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n] # like multiple return statements

json_filepath_chunks = list(chunks(json_filepaths, int(len(json_filepaths) / 15)))
track_ids_set = set(track_ids)

def getTrackTags(sublist):
    tracks_dict = dict()
    progress_bar2 = tqdm(range(len(sublist)))
    for iteration in progress_bar2:
        filepath = sublist[iteration]
        with open(filepath, 'r') as f:
            track_dict = simplejson.load(f)
            # if the track contains tags, add it to my outer dict
            if track_dict['tags'] and track_dict['track_id'] in track_ids_set:
                tracks_dict[track_dict['track_id']] = track_dict['tags']
                # key = track ID, value = list of tags (tags = [string, int])
        progress_bar2.set_description("Iteration %d" % iteration)
    return tracks_dict

import multiprocessing
output = []
with multiprocessing.Pool(16) as pool:
    output = pool.map(getTrackTags, json_filepath_chunks)

# f.close()
print("Len Output: ", len(output))
print("Len(Output[0]): ", len(output[0]))
print("Combining results into single list")
output_list = [tracks_dict for tracks_dict in output]
# I have a list of dicts now I need to combine those

combined_tags_dict = {}
for d in output_list:
    combined_tags_dict = {**combined_tags_dict, **d}

print("Num Tracks with Tags: ", len(combined_tags_dict.keys()))
print("Example tracks with tags dictionary entries: ")
for i in range(3):
    key = random.choice(list(combined_tags_dict))
    print("\nKey and Value:")
    print(key)
    pprint(combined_tags_dict[key])


ordered_tags = [combined_tags_dict.get(id, None) for id in track_ids] # using the track ids list here for the correct ordering

df['tags'] = ordered_tags
print("Exporting dataframe as compressed pickle file...")
# All data is collected, now export as dataframe object
df.to_pickle("dataframe_with_vector_compressed.pkl", compression='zip')
print("COMPLETED!")
