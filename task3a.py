from bs4 import BeautifulSoup
import re
import os
from collections import Counter, OrderedDict
import nltk
import glob
import math
import operator

inverted_unigram_dict = dict()
unigram_termcount = {}
unigram_corpus_count = 0
path = 'corpus'
smoothening_factor = 0.35
commonwords = open('test-collection/common_words', "r")
commonwords = commonwords.read().split('\n')


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


def writeToFile(queryid, queryname, lmscore_dict, folder_name, system_name):
    fo = open("baseline-runs/" + folder_name + "/" + "Q" + str(queryid) + ".txt", "w")
    for key, val in lmscore_dict.items():
        rank = list(lmscore_dict.keys()).index(key) + 1
        #         print(queryid,"\tQ0\t",key,"\t",rank,"\t",val,"LMDirichlet\n")
        fo.write(str(queryid) + "\tQ0\t" + str(key) + "\t" + str(rank) + "\t" + str(val) + "\t" + system_name + "\n")
    fo.close()


# In[2]:


def get_bm25(document, query, document_bm25_score_dict):
    total_score = 0
    score = 0
    query_list = query.split(" ")
    query_list = [x for x in query_list if x not in commonwords]

    average_length = unigram_corpus_count / number_of_docs
    for query_word in query_list:
        R = 0.0
        r = 0.0
        # number of docs containing the term

        n = unigram_invertedlist_count.get(query_word)[0] if unigram_invertedlist_count.get(query_word) else 0

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


def get_tf_idf(document, query, document_tfidfscore_dict):
    total_score = 0
    score = 0
    query_list = query.split(" ")
    query_list = [x for x in query_list if x not in commonwords]

    for query_word in query_list:
        if (getDocWordFreq(query_word, document) != 0 and unigram_termcount.get(document)):
            tf_value = float(getDocWordFreq(query_word, document)) / float(unigram_termcount[document])
        else:
            tf_value = 0
        if (unigram_invertedlist_count.get(query_word)):
            idf_value = math.log(number_of_docs / unigram_invertedlist_count[query_word][0])
        else:
            idf_value = 0

        score = tf_value * idf_value
        total_score += score
    document_tfidfscore_dict[document] = total_score
    return document_tfidfscore_dict


def getJMQLScores(document, query, document_lmscore_dict):
    total_score = 0
    score = 0
    query_list = query.split(" ")
    query_list = [x for x in query_list if x not in commonwords]

    for query_word in query_list:
        if (getDocWordFreq(query_word, document) != 0) and (getCorpusWordFreq(query_word) != 0):
            score = math.log(((1 - smoothening_factor) * float(
                getDocWordFreq(query_word, document) / unigram_termcount[document])) + (
                             smoothening_factor * float(getCorpusWordFreq(query_word) / unigram_corpus_count)))

        else:
            score = 0
        total_score += score
    document_lmscore_dict[document] = total_score
    return document_lmscore_dict


def populate_bm25(queryString):
    document_bm25_score_dict = {}
    docList = []
    query_list = queryString.split(" ")
    query_list = [x for x in query_list if x not in commonwords]
    for query in query_list:
        if unigram_invertedlist_count.get(query):
            query_doclist = unigram_invertedlist_count[query][1]
            for i in query_doclist:
                docList.append(i[0])
        else:
            query_doclist = []

    for docid in docList:
        document_bm25_score_dict = get_bm25(docid, queryString, document_bm25_score_dict)
        document_bm25_score_dict = dict(
            sorted(document_bm25_score_dict.items(), key=operator.itemgetter(1), reverse=True)[:100])
    return document_bm25_score_dict


def populate_tfidf(queryString):
    document_tfidfscore_dict = {}
    docList = []
    query_list = queryString.split(" ")
    query_list = [x for x in query_list if x not in commonwords]
    for query in query_list:
        if unigram_invertedlist_count.get(query):
            query_doclist = unigram_invertedlist_count[query][1]
            for i in query_doclist:
                docList.append(i[0])
        else:
            query_doclist = []

    for docid in docList:
        document_tfidfscore_dict = get_tf_idf(docid, queryString, document_tfidfscore_dict)
        document_tfidfscore_dict = dict(
            sorted(document_tfidfscore_dict.items(), key=operator.itemgetter(1), reverse=True)[:100])
    return document_tfidfscore_dict


def populateJMQL(queryString):
    document_lmscore_dict = {}
    docList = []
    query_list = queryString.split(" ")
    query_list = [x for x in query_list if x not in commonwords]
    for query in query_list:
        if unigram_invertedlist_count.get(query):
            query_doclist = unigram_invertedlist_count[query][1]
            for i in query_doclist:
                docList.append(i[0])
        else:
            query_doclist = []

    for docid in docList:
        document_lmscore_dict = getJMQLScores(docid, queryString, document_lmscore_dict)
        document_lmscore_dict = dict(
            sorted(document_lmscore_dict.items(), key=operator.itemgetter(1), reverse=True)[:100])
    return document_lmscore_dict


# In[7]:


for filename in glob.glob("corpus/*.txt"):
    fo = open(filename, "r")
    data = fo.read()
    tokens = nltk.word_tokenize(data)
    unigramlist = nltk.word_tokenize(data)
    unigramlist = [x for x in unigramlist if x not in commonwords]
    unigram_termcount[os.path.basename(filename).split(".")[0]] = len(unigramlist)
    unigram_corpus_count = unigram_corpus_count + len(unigramlist)
    unigram_dict = Counter(unigramlist)
    inverted_unigram_dict = createIndexDict(filename, unigram_dict, inverted_unigram_dict)
    fo.close()

unigram_invertedlist_count = getInvertedListCount(inverted_unigram_dict)
number_of_docs = len(glob.glob('corpus/*.txt'))

all_queries = query_preprocessor()[1:]

for i in range(len(all_queries)):
    print(i)
    bm25_score_dict = populate_bm25(all_queries[i])
    writeToFile(i+1,all_queries[i],bm25_score_dict,"task3-bm25-stopping","ccisneu_StopNoStem_wordunigram_BM25")
    tfidf_score_dict = populate_tfidf(all_queries[i])
    writeToFile(i + 1, all_queries[i], tfidf_score_dict, "task3-tfidf-stopping", "ccisneu_StopNoStem_wordunigram_TFIDF")
    jmqlscore_dict = populateJMQL(all_queries[i])
    writeToFile(i+1,all_queries[i],jmqlscore_dict,"task3-JMQL-stopping","ccisneu_StopNoStem_wordunigram_JMQL")

