from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import operator


def read_corpus(file_path):
    data = []
    for line in open(file_path):
        sent = line.strip().lower().split(' ')
        # only append <s> and </s> to the target sentence
        sent = ['START'] + sent + ['END']
        data.append(sent)

    return data

def load_ngram(file_path,vocab,n=4):
    data=read_corpus(file_path)
    return load_ngram_without_corpus(data,vocab,n)


def load_ngram_without_corpus(data_list,vocab,n=4):
    ngram = []

    for sent in data_list:
        for i in range(0, len(sent) + 1 - n):
            ngram += [[vocab.word_to_id(token) for token in sent[i:i + 4]]]
    return np.asarray(ngram)


def plot_ngram(data, vocab=None, n=4, top=50):
    ngram = []

    for sent in data:
        for i in range(0, len(sent) + 1 - n):
            if vocab:
                ngram += [" ".join([token if (not vocab.word_to_id(token) == 0) else "UNK" for token in sent[i:i + 4]])]
            else:
                ngram += [" ".join(sent[i:i + 4])]

    count_ngram = Counter(ngram)

    sorted_freqs = sorted(count_ngram.iteritems(), key=operator.itemgetter(1), reverse=True)

    labels, values = zip(*(sorted_freqs))

    indexes = np.arange(len(labels))

    plt.scatter(indexes, values, s=1)
    plt.title("Distribution of " + str(n) + "-gram and their counts")
    plt.grid()
    plt.show()
    plt.cla()

    print "top 50 ngrams"
    print sorted_freqs[:top]

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    try:
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    except:
        print "exception softmax"
        print x
        sys.exit(0)




# def plot_ngram_vocab(data, vocab, n=4, top=50):
#     ngram = []
#
#
#     count_ngram = Counter(ngram)
#
#     sorted_freqs = sorted(count_ngram.iteritems(), key=operator.itemgetter(1), reverse=True)[:top]
#
#     labels, values = zip(*(sorted_freqs))
#
#     indexes = np.arange(len(labels))
#
#     plt.scatter(indexes, values, s=1)
#     plt.title("Distribution of " + str(n) + "-gram and their counts")
#     plt.grid()
#     plt.show()
#     plt.cla()
#
#     print "top 50 ngrams"
#     print sorted_freqs[:top]
