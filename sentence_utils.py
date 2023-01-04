import numpy as np
from math import log
import re


def clean_quot(article):
    quot = ["\"", "'"]
    for i in range(len(quot)):
        article.replace(quot[i], "")
    return article


# most lengthy sentence
def get_max_len_sent(article):
    # split an article into sentences
    sents = re.split("[.?!]", article)
    # traverse through list of sentences
    max_len_sent = 0
    for sent in sents:
        # split a sentence into a list of words for counting
        sent = sent.split()
        if len(sent) > max_len_sent:
            max_len_sent = len(sent)
    return max_len_sent


# num of sentences
def get_num_sents(article):
    sents = re.split("[.?!]", article)
    return len(sents)


# avg length of sentences
def get_avg_len_sents(article):
    # get average length of sentences
    sents = re.split("[.?!]", article)
    total_len_sents = 0
    for sent in sents:
        total_len_sents += len(sent.split())
    avg_len_sents = total_len_sents / len(sents)
    return avg_len_sents

def get_max_len_word(article):
    _, vocab, _ = get_words(article)
    max_len_word = 0
    for i in range(len(vocab)):
        if len(vocab[i]) > max_len_word:
            max_len_word = len(vocab[i])
    return max_len_word


def get_num_words(article):
    word_list, vocab, _ = get_words(article)
    num_words = len(word_list)
    num_vocab = len(vocab)
    return num_words, num_vocab


def get_avg_len_words(word_list):
    total_len_words = 0
    for word in word_list:
        total_len_words += len(word)
    avg_len_words = total_len_words / len(word_list)
    return avg_len_words


def get_words(article):
    word_list = article.split()
    punctuation = [",", ".", ":", "\"", "'", ";", "?", "(", ")", "!"]
    for i in range(len(word_list)):
        for j in range(len(punctuation)):
            word_list[i] = word_list[i].replace(punctuation[j], "")
    for i in range(len(word_list)):
        word_list[i] = word_list[i].lower()
    vocab = list(set(word_list))
    word_hist = np.zeros(len(vocab))
    for i in range(len(word_list)):
        for j in range(len(vocab)):
            if word_list[i] == vocab[j]:
                word_hist[j] += 1
    word_hist = word_hist / word_hist.sum()
    return word_list, vocab, word_hist


def clean_vocab(vocab):
    stop_symbols = ["@", "…", "“", "”", "‘", "<", ">", "&", "[", "]", "/", "\\"]
    new_vocab = []
    for word in vocab:
        word = word.strip("-")
        if word.isnumeric():
            continue
        flag = True
        for stop_symbol in stop_symbols:
            if stop_symbol in word:
                flag = False
        if flag:
            new_vocab.append(word)
    return new_vocab



# abundance of vocab
def article_entropy(word_hist):
    entropy = 0
    for i in range(word_hist.shape[0]):
        entropy += (-word_hist[i] * log(word_hist[i], 2))
    return entropy * word_hist.shape[0]


if __name__ == "__main__":
    article = """
        The tower is 324 metres (1,063 ft) tall, about the same height as an 
        81-storey building, and the tallest structure in Paris. Its base is 
        square, measuring 125 metres (410 ft) on each side. During its 
        construction, the Eiffel Tower surpassed the Washington Monument to 
        become the tallest man-made structure in the world, a title it held for 
        41 years until the Chrysler Building in New York City was finished in 
        1930. It was the first structure to reach a height of 300 metres. Due 
        to the addition of a broadcasting aerial at the top of the tower in 
        1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). 
        Excluding transmitters, the Eiffel Tower is the second tallest 
        free-standing structure in France after the Millau Viaduct.
    """


    # print("max_len of sentence: ", get_max_len_sent(article))
    # print("num of sentences: ", get_num_sents(article))
    # print("average len of sentences: ", get_avg_len_sents(article))
    # word_list, vocab, word_hist = get_words(article)
    # entropy = article_entropy(word_hist)
    # print(word_list)
    # print(vocab)
    # print(word_hist)
    # print(entropy)
    # print(get_max_len_word(article))
    # print(clean_quot(article))
    # print(get_num_words(article))
    # print(get_avg_len_words(word_list))
