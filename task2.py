
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


def populate_bm25(queryString):
    document_bm25_score_dict = {}
    docList = []
    query_list = queryString.split(" ")

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



def pseudo_relavance(query, bm25_score_dict):
    k = 10
    # Generate query vector
    query_vector = {}
    query_list = query.split(" ")

    for query_word in query_list:
        if (query_vector.get(query_word)):
            query_vector[query_word] += 1
        else:
            query_vector[query_word] = 1

    for key, val in unigram_invertedlist_count.items():
        if not query_vector.get(key):
            query_vector[key] = 0
    relevance_vector, relevance_vector_magnitude = generate_relevance_vector(query, bm25_score_dict, k)

    non_relevance_vector, non_relevance_vector_magnitude = generate_non_relevance_vector(query, bm25_score_dict, k)

    expanded_q = query_expansion(query, query_vector, relevance_vector, relevance_vector_magnitude,
                                 non_relevance_vector, non_relevance_vector_magnitude)

    return (expanded_q)


def generate_relevance_vector(query, bm25_score_dict, k):
    sorted_doc = dict(sorted(bm25_score_dict.items(), key=operator.itemgetter(1), reverse=True)[:k])
    relevance_vector = {}
    for key, val in sorted_doc.items():
        with open('corpus/' + key + '.txt', 'r') as f:
            doc_list = f.read().split()
            for term in doc_list:
                if (relevance_vector.get(term)):
                    relevance_vector[term] += 1
                else:
                    relevance_vector[term] = 1

            for token, val in unigram_invertedlist_count.items():
                if not relevance_vector.get(token):
                    relevance_vector[token] = 0

    relevance_vector_magnitude = 0
    for key, val in relevance_vector.items():
        relevance_vector_magnitude += float(val ** 2)

    relevance_vector_magnitude = math.sqrt(relevance_vector_magnitude)
    return (relevance_vector, relevance_vector_magnitude)



def generate_non_relevance_vector(query, bm25_score_dict, k):
    sorted_doc = dict(sorted(bm25_score_dict.items(), key=operator.itemgetter(1), reverse=True)[k:])
    non_relevance_vector = {}
    for key, val in sorted_doc.items():
        with open('corpus/' + key + '.txt', 'r') as f:
            doc_list = f.read().split()
            for term in doc_list:
                if (non_relevance_vector.get(term)):
                    non_relevance_vector[term] += 1
                else:
                    non_relevance_vector[term] = 1

            for token, val in unigram_invertedlist_count.items():
                if not non_relevance_vector.get(token):
                    non_relevance_vector[token] = 0

    non_relevance_vector_magnitude = 0
    for key, val in non_relevance_vector.items():
        non_relevance_vector_magnitude += float(val ** 2)

    non_relevance_vector_magnitude = math.sqrt(non_relevance_vector_magnitude)
    return (non_relevance_vector, non_relevance_vector_magnitude)


def query_expansion(query, query_vector, relevance_vector, relevance_vector_magnitude, non_relevance_vector,
                    non_relevance_vector_magnitude):
    query_expansion_dict = {}
    for term, val in unigram_invertedlist_count.items():
        query_expansion_dict[term] = query_vector[term] + (0.5 / relevance_vector_magnitude) * relevance_vector[
            term] - (0.15 / non_relevance_vector_magnitude) * non_relevance_vector[term]

    query_expansion_dict = dict(sorted(query_expansion_dict.items(), key=operator.itemgetter(1), reverse=True))

    expanded_query = query
    no_extra_query_terms = 15
    for i in range(no_extra_query_terms):
        term = list(query_expansion_dict.keys())[i]
        if term not in query:
            expanded_query += " " + term
    return expanded_query


def get_all_new_queries():
    for old_query in range(len(all_queries)):
        expanded_q = pseudo_relavance(all_queries[old_query], bm25_score_dict)
        expanded_query_list.append(expanded_q)

    return expanded_query_list



def writeToFile(queryid, queryname, lmscore_dict, folder_name, system_name):
    fo = open("baseline-runs/" + folder_name + "/" + "Q" + str(queryid) + ".txt", "w")
    for key, val in lmscore_dict.items():
        rank = list(lmscore_dict.keys()).index(key) + 1
        #         print(queryid,"\tQ0\t",key,"\t",rank,"\t",val,"LMDirichlet\n")
        fo.write(str(queryid) + "\tQ0\t" + str(key) + "\t" + str(rank) + "\t" + str(val) + "\t" + system_name + "\n")
    fo.close()


expanded_query_list = []
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
number_of_docs = len(glob.glob('corpus/*.txt'))

all_queries = query_preprocessor()[1:]

for i in range(len(all_queries)):
    print(all_queries[i])
    bm25_score_dict = populate_bm25(all_queries[i])


all_expanded_queries = get_all_new_queries()

for i in range(len(all_expanded_queries)):
    bm25_score_dict_expanded = populate_bm25(all_expanded_queries[i])
    writeToFile(i + 1, all_queries[i], bm25_score_dict_expanded, "task2-bm25-pseudo-relevance",
                "ccisneu_wordunigram_BM25_PSEUDO_RELEVANCE")

