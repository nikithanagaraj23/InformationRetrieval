from bs4 import BeautifulSoup
import os
from collections import Counter
import nltk
import glob
import functools
import itertools
import re
import pandas as pd
import operator
import math
from tqdm import tqdm_notebook as tqdm
pd.set_option('display.max_colwidth', -1)

inverted_unigram_dict = dict()
unigram_termcount = {}
unigram_corpus_count = 0
path = 'corpus'
smoothening_factor = 0.35


def casefolding(data):
    return data.lower()


def punctuationHandling(data):
    regex = r"(?<!\d)[.,;:*!\"\'#$%&()+/<=>?@[\]_^`~{|}∑α](?!\d)"
    data = re.sub(regex, "", data, 0)
    data = re.sub(r'\d+', '', data)
    regex = r"[;:*!\"\'#$%&()+/<=>?@[\]_^`~{|}]"
    data = re.sub(regex, "", data, 0)
    return data


def removeWhitespace(data):
    data = ' '.join(data.split())
    return data


def createCorpusFile(heading, maincontent):
    fo = open("corpus/" + str(heading) + ".txt", "w")
    fo.write(maincontent)
    fo.close()


def parseDocs():
    for filename in glob.glob("test-collection/cacm/*.html"):
        fo = open(filename, "r")
        heading = os.path.basename(filename).split(".")[0]
        data = fo.read()
        maincontent = BeautifulSoup(data, "lxml").text
        maincontent = casefolding(maincontent)
        maincontent = punctuationHandling(maincontent)
        maincontent = removeWhitespace(maincontent)
        createCorpusFile(heading, maincontent)


def createIndexDict(file, ngram_dict, inverted_dict):
    for key, value in ngram_dict.items():
        if (inverted_dict.get(key)):
            inverted_dict.get(key).append((os.path.basename(file).split(".")[0], value))
        else:
            inverted_dict[key] = [(os.path.basename(file).split(".")[0], value)]
    return inverted_dict


def getInvertedListCount(index_list):
    invertedlist_count = {}
    for key, val in index_list.items():
        invertedlist_count[key] = [len(val), val]
    return invertedlist_count


def getDocWordFreq(word, document):
    value = 0
    if unigram_invertedlist_count.get(word):
        for val in unigram_invertedlist_count[word][1]:
            if (val[0] == document):
                value = val[1]
    return value


def getCorpusWordFreq(word):
    if unigram_invertedlist_count.get(word):
        return unigram_invertedlist_count[word][0]
    else:
        return 0


def query_preprocessor(filepath='test-collection/cacm.query.txt'):
    with open(filepath) as f: queries = f.read()
    queries = [l.replace('</DOCNO>', '').replace('\n', ' ').replace('</DOC>', '').replace('<DOC>', '')[1:] for l in
               queries.split('<DOCNO>')]
    queries = [re.sub(r'^\d*\s\s', '', l) for l in queries]
    queries = [s.lower() for s in queries]
    queries = [punctuationHandling(query) for query in queries]
    queries = [removeWhitespace(query) for query in queries]
    return queries



def get_bm25(document, query, document_bm25_score_dict):
    total_score = 0
    score = 0
    query_list = query.split(" ")

    average_length = unigram_corpus_count / number_of_docs
    for query_word in query_list:
        R = 0.0
        r = 0.0
        # number of docs containing the term

        n = unigram_invertedlist_count.get(query_word)[0] if unigram_invertedlist_count.get(query_word) else 0
        #         n = unigram_invertedlist_count.get(query_word)[0]


        # Total number of documents
        N = number_of_docs
        k1 = 1.2
        k2 = 100
        # freq of the word in query
        qf = query_list.count(query_word)

        # freq of word in the doc
        f = getDocWordFreq(query_word, document)

        # to be calculated using b= 0.75
        b = 0.75
        K = k1 * ((1 - b) + b * (float(unigram_termcount[document]) / float(average_length)))
        smoothening = 0.5

        first_part = math.log(((r + 0.5) / (R - r + 0.5)) / ((n - r + 0.5) / (N - n - R + r + 0.5)))
        second_part = ((k1 + 1) * f) / (K + f)
        third_part = ((k2 + 1) * qf) / (k2 + qf)
        score = first_part * second_part * third_part

        total_score += score
    document_bm25_score_dict[document] = total_score
    return document_bm25_score_dict


def populate_bm25(queryString, docList):
    document_bm25_score_dict = {}
    query_list = queryString.split(" ")
    for docid in docList:
        document_bm25_score_dict = get_bm25(docid, queryString, document_bm25_score_dict)
        document_bm25_score_dict = dict(
            sorted(document_bm25_score_dict.items(), key=operator.itemgetter(1), reverse=True)[:100])
    return document_bm25_score_dict



def positionIndex(word_list, docid, position_index_dict, visitedfiles, completedfiles):
    for word in word_list:
        indices = [i for i, x in enumerate(word_list) if x == word]
        gaps = [j - i for i, j in zip(indices[:-1], indices[1:])]
        gaps = [indices[0]] + gaps

        if (position_index_dict.get(word)) != None and (docid not in completedfiles):
            newdata = (docid, len(gaps), gaps)
            if not newdata in position_index_dict.get(word):
                position_index_dict.get(word).append((docid, len(gaps), gaps))

        else:
            if not position_index_dict.get(word):
                position_index_dict[word] = [(docid, len(gaps), gaps)]
                if docid not in visitedfiles:
                    visitedfiles.append(docid)

    return visitedfiles, completedfiles, position_index_dict



def writeToFile(queryid, queryname, lmscore_dict, folder_name, system_name):
    fo = open("extra-credits/" + folder_name + "/" + "Q" + str(queryid) + ".txt", "w")
    for key, val in lmscore_dict.items():
        rank = list(lmscore_dict.keys()).index(key) + 1
        fo.write(str(queryid) + "\tQ0\t" + str(key) + "\t" + str(rank) + "\t" + str(val) + "\t" + system_name + "\n")
    fo.close()

def positionIndex(word_list, docid, position_index_dict, visitedfiles, completedfiles):
    for word in word_list:

        indices = [i for i, x in enumerate(word_list) if x == word]

        if word in position_index_dict:
            if docid in position_index_dict[word]:
                position_index_dict[word][docid] = indices
            else:
                position_index_dict[word].update({docid: indices})

        else:
            position_index_dict.update({word: {docid: indices}})

    return position_index_dict




def createPositionIndex():
    position_index_dict = {}
    visitedfiles = []
    completedfiles = []

    for filename in tqdm(glob.glob("corpus/*.txt")):
        fo = open(filename, "r")
        data = fo.read()
        tokens = nltk.word_tokenize(data)
        unigramlist = nltk.word_tokenize(data)

        position_index_dict = positionIndex(unigramlist,
                                            (os.path.basename(filename).split(".")[0]),
                                            position_index_dict, visitedfiles, completedfiles)

        completedfiles.append((os.path.basename(filename).split(".")[0]))

    return position_index_dict


def check_sequence(rel_docs, query_words, position_index_dict):
    check_list = []

    prev_word = query_words[0]

    for doc_id in rel_docs:

        doc_flag = True

        for word in query_words:

            if not min(position_index_dict[prev_word][doc_id]) <= max(position_index_dict[word][doc_id]):
                doc_flag = False

            prev_word = word

        if doc_flag:
            check_list.append(True)
        else:
            check_list.append(False)

    return [doc for i, doc in enumerate(rel_docs) if check_list[i]]


def exact_match(query, position_index_dict, check_seq_flag=False):
    query_words = query.split(' ')
    return_doc_list = []

    for word in query_words:

        if word in position_index_dict:

            return_doc_list.append(set(position_index_dict[word].keys()))
        else:
            return []

    rel_docs = functools.reduce(lambda s1, s2: s1 & s2, return_doc_list)

    if check_seq_flag:
        rel_docs = check_sequence(rel_docs, query_words, position_index_dict)

    return rel_docs


def best_match(query, position_index_dict, check_seq_flag=False):
    query_words = query.split(' ')
    return_doc_list = []

    for word in query_words:

        if word in position_index_dict:
            return_doc_list.extend(position_index_dict[word].keys())

    return set(return_doc_list)


def check_proximity(rel_docs, query_pairs, position_index_dict, N):
    check_list = []

    for doc_id in rel_docs:

        doc_flag = False

        for word_pairs in query_pairs:

            if (doc_id in position_index_dict[word_pairs[1]]) and (doc_id in position_index_dict[word_pairs[0]]):

                min_distance = 9999999999

                word_1_pos = position_index_dict[word_pairs[0]][doc_id]
                word_2_pos = position_index_dict[word_pairs[1]][doc_id]

                for pos_1 in word_1_pos:

                    for pos_2 in word_2_pos:

                        if ((pos_2 - pos_1) > 0) and ((pos_2 - pos_1) <= N):
                            doc_flag = True

        if doc_flag:
            check_list.append(True)
        else:
            check_list.append(False)

    return [doc for i, doc in enumerate(rel_docs) if check_list[i]]


def proximity_match(query, position_index_dict, N=10):
    query_words = query.split(' ')
    query_words = [q for q in query_words if q in position_index_dict.keys()]
    return_doc_list = []

    for word in query_words:

        if word in position_index_dict:
            return_doc_list.extend(position_index_dict[word].keys())

    return_doc_list = list(set(return_doc_list))

    return check_proximity(return_doc_list,
                           list(itertools.combinations(query_words, 2)),
                           position_index_dict, N)


# In[265]:


for filename in glob.glob("corpus/*.txt"):
    fo = open(filename, "r")
    data = fo.read()
    tokens = nltk.word_tokenize(data)
    unigramlist = nltk.word_tokenize(data)
    unigram_termcount[os.path.basename(filename).split(".")[0]] = len(unigramlist)
    unigram_corpus_count = unigram_corpus_count + len(unigramlist)
    unigram_dict = Counter(unigramlist)
    inverted_unigram_dict = createIndexDict(filename, unigram_dict, inverted_unigram_dict)
    fo.close()

unigram_invertedlist_count = getInvertedListCount(inverted_unigram_dict)
position_index_dict = createPositionIndex()
number_of_docs = len(glob.glob('corpus/*.txt'))

all_queries = query_preprocessor()[1:]

# for i in range(len(all_queries)):
#     print(i+1)
#     docList = exact_match(all_queries[i], position_index_dict, True)
#     docList = best_match(all_queries[i], position_index_dict, True)
#     docList = proximity_match(all_queries[i], position_index_dict, 10)
#     bm25_score_dict = populate_bm25(all_queries[i],docList)
#     print(bm25_score_dict)
#     writeToFile(i+1,all_queries[i],bm25_score_dict,"exact-match","ccisneu_wordunigram_ExactMatch")
#     writeToFile(i+1,all_queries[i],bm25_score_dict,"ordered-best-match","ccisneu_wordunigram_OrderedBestMatch")


# In[275]:

def printOutput(queryid,queryname,lmscore_dict,folder_name,system_name):
    for key,val in lmscore_dict.items():
        rank = list(lmscore_dict.keys()).index(key)+1
        print(queryid,"\tQ0\t",key,"\t",rank,"\t",val,"\t",system_name,"\n")

function_name = input(
    "Enter the function number you would want to use:\n1.Exact Match\t2.Best Match \t3. Ordered Best Match with proximity\n")
if function_name == '1':
    for i in range(len(all_queries)):
        docList = exact_match(all_queries[i], position_index_dict, True)
        bm25_score_dict = populate_bm25(all_queries[i], docList)
        printOutput(i + 1, all_queries[i], bm25_score_dict, "Exact-match", "ccisneu_wordunigram_ExactMatch")
elif function_name == '2':
    for i in range(len(all_queries)):
        docList = best_match(all_queries[i], position_index_dict, True)
        bm25_score_dict = populate_bm25(all_queries[i], docList)
        printOutput(i + 1, all_queries[i], bm25_score_dict, "best-match", "ccisneu_wordunigram_BestMatch")
elif function_name == '3':
    N = int(input("Enter the proximity window\n"))
    for i in range(len(all_queries)):
        print(all_queries[i])
        docList = proximity_match(all_queries[i], position_index_dict, N)
        bm25_score_dict = populate_bm25(all_queries[i], docList)
        printOutput(i + 1, all_queries[i], bm25_score_dict, "ordered-best-match", "ccisneu_wordunigram_OrderedBestMatch")

