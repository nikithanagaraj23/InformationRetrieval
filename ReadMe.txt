SETUP and Program Execution:
* Open the main folder on any of the ides that support python like pycharm.This folder contains all the programs for Phase1,Phase2 and Phase3.
* The program uses python 3.
* To run the code either run the code using any ide which supports python like pycharm, or run on terminal using
python file.py

Content:
This folder contains the below mentioned files and folders:
* test-collection/ -  this is the test data provided to us to run all our programs
* baseline-runs/ - the folder that contains the outputs of all the runs from task1, task2 , task3 from phase1 and phase2
* corpus/ -  this folder holds the cleaned version of the cacm corpus provided to us in the test-collection. This was generated
    as a part of task1 of phase1.The data in this corpus holds text without numbers and punctuations. All the data is in lower case and
    extra spaces have been removed.
* corpus_stem/ - this folder holds the cleaned verion of the cacm_stem.txt corpus provided in test-collection.This was generated
as a part of task3 of phase1. The file was parsed and each document was added to a separate file appended with CACM such that the data in
corpus_stem/ ranges from CACM-0001 to CACM-3204.
* phase2-output/ -  this folder contains the output from phase2. The snippets have been generated to bm25 retrieval model. Under the folder
snippets -  each of the files named based on the Query id have been created. These files have the document id and the top three sentences
which best describe the document for that query. The query word appearing in this file have been made bold.
* reusable_data/ - this folder contains the inverted index and the inverted index with the count of documents. It also contains the queryid to
query mapping which is used in phase2.
*




