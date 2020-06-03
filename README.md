# cs4641-project group9
ML Group Project Spring 2020

Please install GIT LFS before cloning to repo to make sure that the large file(s) are pulled successfully.

[GIT LFS Installation Guide](https://help.github.com/en/github/managing-large-files/installing-git-large-file-storage). Make sure you follow the steps corresponding to your operation system.

In order to load in the full, simplified data as a `pandas` DataFrame object, run the following code in python:

    import pandas as pd
    df = pd.read_pickle('dataframe_with_vector_compressed.pkl', compression='zip')

You can print / view the DataFrame object `df` after that.

Each column has a title that tells you what value you should find there. You can see the column titles by printing `df.columns`.

To get the actual words that correspond to the index in the bag of words:

    import json
    with open("word_list.json", 'r') as w:
        word_list = list(json.load(w))

This will load the list of words, and should be of length 5001.

Note index 0 in the `word_list` is whitespace. This is intentional.

With this list, you can simply plug in the index from the bag of words to see what word it corresponds to.


If you want to use or modify the data_cleanup_script.py that aggregates the initial data, you will need to download the files from the Million Song Dataset at the following links:
1. Summary HD5 file: http://millionsongdataset.com/sites/default/files/AdditionalFiles/msd_summary_file.h5 
2. Lyrics Training file: http://millionsongdataset.com/sites/default/files/AdditionalFiles/mxm_dataset_train.txt.zip
3. Lyrics Testing file: http://millionsongdataset.com/sites/default/files/AdditionalFiles/mxm_dataset_test.txt.zip
4. Genre Tags Training file: http://millionsongdataset.com/sites/default/files/lastfm/lastfm_train.zip
5. Genre Tags Testing file: http://millionsongdataset.com/sites/default/files/lastfm/lastfm_test.zip

Once you have these files and unzipped them in the same directory as data_cleanup_script.py, you should be able to run the script.
