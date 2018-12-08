SETUP and Program Execution:
* Open the main folder on any of the ides that support python like pycharm.This folder contains all the programs for Phase1,Phase2 and Phase3.
* The program uses python 3.
* To run the code either run the code using any ide which supports python like pycharm, or run on terminal using
python file.py (All except Phase 1 Task1 - Lucene)

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
* task1/task1.py - This file contains the implementation of corpus creation, indexing and the retrieval models :
BM25, tf-idf, and JM Smoothed Query Likelihood Model . The Outputs for these runs are present in the folder baseline-runs:
    baseline-runs/task1-bm25 - The top 100 ranked documents for BM25 retrieval model.
    baseline-runs/task1-JMQL - The top 100 ranked documents for JM Smoothed Query Likelihood Model retrieval model
    baseline-runs/task1-tfidf - The top 100 ranked documents for  tf-idf retrieval model
* task1/lucene.java - this file contains the implementation of the task1 of phase 1 for retrieval using Lucene. For this we just need
 to open the java file in any of the ide and run the file. This will ask you for the location where the index need to be created
 and the location where the raw corpus is present. This will generate the lucene evaluations for the queries. The Output of the program is
 present in baseline-runs/task1-lucene_evaluations
* task2.py - this file contains the implementation of task2 of phase1. We have applied Psuedo Relevance query encrichment on
BM25 retrieval model. The output of this program is stored in task2/
* task3a.py - this file contains the implementation of task3a of phase1. The output of these runs are places in the folder baseline-runs:
    baseline-runs/task1-bm25-stopping - The top 100 ranked documents for BM25 retrieval model after removing stop words.
    baseline-runs/task1-JMQL-stopping - The top 100 ranked documents for JM Smoothed Query Likelihood Model retrieval model after removing stop words.
    baseline-runs/task1-tfidf-stopping - The top 100 ranked documents for  tf-idf retrieval model after removing stop words.
* task3b.py - this file contains the implementation of task3b of phase1. The output of these runs are places in the folder baseline-runs:
    baseline-runs/task1-bm25-stemming - The top 100 ranked documents for BM25 retrieval model using corpus_stem and stem query
    baseline-runs/task1-JMQL-stemming - The top 100 ranked documents for JM Smoothed Query Likelihood Model retrieval modelus using
    corpus_stem and stem query.
    baseline-runs/task1-tfidf-stemming - The top 100 ranked documents for  tf-idf retrieval model using corpus_stem and stem query.
* phase2.py - this file is the implementation of Phase2. The snippets have been generated and placed into the folder
phase2-output/snippets/
* extra_credit.py - this program implements the extra credit section.To run the file you can either use an ide or the command line. This will ask
you to select one of the three retrieval mechanisms. On selecting the number the respective function will be executed and the
query with results will be displayed. We have run these three functions for the already existing queries and the outputs have been placed in
extra-credits/ folder.
* extra-credits/ -  holds the results of extra-credits.py