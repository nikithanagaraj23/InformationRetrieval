
import re
from bs4 import BeautifulSoup
import pickle
import pandas as pd
import operator
from nltk.tokenize import word_tokenize
from os import listdir
from os.path import isfile, join
from tqdm import tqdm_notebook as tqdm

pd.set_option('display.max_colwidth', -1)


def casefolding(data):
    return data.lower()


def punctuationHandling(data):
    regex = r"(?<!\d)[.,;:*!\"\'#$%&()+/<=>?@[\]_^`~{|}∑α](?!\d)"
    data = re.sub(regex, "", data, 0)
    data = re.sub(r'\d+', '', data)
    regex = r"[;:*!\"\'#$%&()+/<=>?@[\]_^`~{|}]"
    data = re.sub(regex, "", data, 0)
    #     data = data.strip(string.punctuation)
    return data


def get_sentences(result):
    doc = open("test-collection/cacm/" + result + ".html", "r")
    data = doc.read()
    data = BeautifulSoup(data, "lxml").text
    data = casefolding(data)
    data = re.sub("\d+", "", data)
    data = re.sub("\n", ".\n", data)
    sentences = re.split(r"\.[\s\n]+", data)
    sentences = [s.replace("\n", " ").replace("\t", " ") for s in sentences]
    sentences = [re.sub(' +', ' ', s) for s in sentences]
    sentences = [punctuationHandling(s) for s in sentences]
    sentences = [x for x in sentences if x]
    return sentences


def getDocWordFreq(word, document):
    value = 0
    if unigram_invertedlist_count.get(word):
        for val in unigram_invertedlist_count[word][1]:
            if (val[0] == document):
                value = val[1]
    return value


def calculate_significant_words(sentences, document, query_words, inverted_index):
    significant_words = []
    words = []
    sd = len(sentences)

    if (sd < 25):
        thresh = 4 - 0.1 * (25 - sd)

    elif ((25 <= sd) and (sd <= 40)):
        thresh = 4

    else:
        thresh = 4 + 0.1 * (sd - 40)

    for sentence in sentences:
        words.extend(word_tokenize(sentence))

    for word in words:
        f_dw = getDocWordFreq(word, document)
        if f_dw >= thresh: significant_words.append(word)

    significant_words.extend(query_words)
    significant_words = list(set(significant_words))

    return significant_words


def calculate_significance_factor(sentences, significant_words):
    word_count = 0
    significance_factor = {}

    for sentence in sentences:
        tokenized_sentence = word_tokenize(sentence)
        filtered_significance_words = set(tokenized_sentence).intersection(set(significant_words))
        min_index = 100000;
        max_index = -1;

        filtered_significance_words = [x for x in filtered_significance_words if x not in commonwords]

        for token in tokenized_sentence:
            # print("token",token)
            if token in filtered_significance_words:

                new_max_index = max([i for i, x in enumerate(tokenized_sentence) if x == token])
                new_min_index = min([i for i, x in enumerate(tokenized_sentence) if x == token])

                if (min_index > new_min_index):
                    min_index = new_min_index

                if (max_index < new_max_index):
                    max_index = new_max_index

                text_span = (max_index - min_index) + 0.0001

                count = sum([1 for w in tokenized_sentence[min_index: max_index] if w in filtered_significance_words])
                significance_factor[sentence] = count ** 2 / text_span

    significance_factor = dict(sorted(significance_factor.items(), key=operator.itemgetter(1), reverse=True)[:3])
    return significance_factor


# In[60]:


def get_text_summary(results, query):
    query_words = query.split(' ')
    query_words = [x for x in query_words if x not in commonwords]
    significance_df = pd.DataFrame()

    for result in results:
        significance_dict = []
        sentences = get_sentences(result)
        significant_words = calculate_significant_words(sentences,
                                                        result, query_words,
                                                        inverted_index)

        for sent, score in calculate_significance_factor(sentences, significant_words).items():

            for query_word in query_words:
                sent = sent.replace(query_word, '<b> ' + query_word + '</b>')

            significance_dict.append({'result': result,
                                      'significance_factor': score,
                                      'sentence': sent})
        significance_df = significance_df.append(pd.DataFrame.from_dict(significance_dict))
    return significance_df


def get_query_id_map():
    with open('./test-collection/cacm.query.txt') as f:
        queries = f.read()

    query_ids = re.findall(r'<DOCNO> \d+ </DOCNO>', queries)
    query_ids = [re.findall(r'\d+', q)[0] for q in query_ids]

    queries = re.split(r'</DOC>', queries)
    queries = [
        l.replace('</DOCNO>', '').replace('\n', ' ').replace('</DOC>', '').replace('<DOC>', '').replace('<DOCNO>', '')
        for l in queries]

    # queries = [re.sub(r'^\d*\s\s', '',l) for l in queries]

    queries = [re.sub(r'\s{2,5}', '', l) for l in queries]
    queries = [re.sub(r'\d{1,2}', '', l) for l in queries]

    queries = pd.DataFrame({'query_ids': query_ids,
                            'queries': queries[:-1]}).set_index('query_ids')

    return query_ids, queries


def iterate_queries(queries, path='baseline-runs/task1-bm25'):
    queries_ids = [f.split('.')[0].replace('Q', '') for f in listdir(path) if isfile(join(path, f))]
    queries_ids = [x for x in queries_ids if x]

    print(len(queries_ids))
    for query_id in tqdm(queries_ids):
        #         print("Queryid",query_id)
        query_results = pd.read_csv(path + 'Q' + query_id + '.txt', sep='\t', names=['qid', 'Q0', 'doc_id',
                                                                                     'rank', 'score', 'system'])

        get_text_summary(list(query_results['doc_id']), queries.loc[queries.index == query_id, 'queries'][query_id])[
            ['result', 'sentence']].to_html('phase2-output/snippets/' + query_id + '.html',
                                            index=False, escape=False, border=0, max_rows=None, max_cols=None)

    return


unigram_invertedlist_count = {}
inverted_index = {}
commonwords = []
query_ids, queries = get_query_id_map()
queries.to_csv('queries.csv')
with open('reusable_data/unigram_invertedlist_count.pkl', 'rb') as f:
    unigram_invertedlist_count = pickle.load(f)

with open('reusable_data/inverted_index.pkl', 'rb') as f:
    inverted_index = pickle.load(f)
commonwords = open('test-collection/common_words', "r")
commonwords = commonwords.read().split('\n')

iterate_queries(queries, path='baseline-runs/task1-bm25/')

