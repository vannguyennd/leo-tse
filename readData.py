import numpy as np
from collections import Counter
import pandas as pd

MAX_NUM_WORDS = 1000000
texts = []


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    np.random.seed(123)
    p = np.random.permutation(num)
    return [d[p] for d in data]


def split_data(data_non_wi, data_non_wi_len, labels_non, data_vul_wi, data_vul_wi_len, labels_vul):
    data_non_wi, data_non_wi_len, labels_non = shuffle_aligned_list([data_non_wi, data_non_wi_len, labels_non])
    data_vul_wi, data_vul_wi_len, labels_vul = shuffle_aligned_list([data_vul_wi, data_vul_wi_len, labels_vul])

    percent = 0.8
    non_size = int(data_non_wi.shape[0] * percent)
    vul_size = int(data_vul_wi.shape[0] * percent)

    x_train = np.concatenate((data_non_wi[:non_size], data_vul_wi[:vul_size]))
    x_train_len = np.concatenate((data_non_wi_len[:non_size], data_vul_wi_len[:vul_size]))
    y_train = np.concatenate((labels_non[:non_size], labels_vul[:vul_size]))

    x_test = np.concatenate((data_non_wi[non_size:], data_vul_wi[vul_size:]))
    x_test_len = np.concatenate((data_non_wi_len[non_size:], data_vul_wi_len[vul_size:]))
    y_test = np.concatenate((labels_non[non_size:], labels_vul[vul_size:]))

    x_train, x_train_len, y_train = shuffle_aligned_list([x_train, x_train_len, y_train])
    x_test, x_test_len, y_test = shuffle_aligned_list([x_test, x_test_len, y_test])

    return x_train, x_train_len, y_train, x_test, x_test_len, y_test


def cwe_ood_fetch_codes_data_vul_non(file_path):
    labels_non = []
    codes_non = []
    labels_vul = []
    codes_vul = []

    df = pd.read_csv(file_path)
    data = []
    for idx in range(df.shape[0]):
        code = df.iloc[idx]['func_rename']
        label = df.iloc[idx]['vul']
        data.append((code, label))
    
    for x in data:
        code_x, label_x = x
        sentences = []

        for i_code in code_x.split("\n")[:-1]:
            sentences.append(i_code.strip())
        if len(sentences) == 0:
            print("ERROR!!!")
        
        if label_x == 1:
            labels_vul.append(label_x)
            codes_vul.append(sentences)
        else:
            labels_non.append(label_x)
            codes_non.append(sentences)

    return codes_non, labels_non, codes_vul, labels_vul


def cwe_ood_fetch_words_data_non_vul(file_path):
    df = pd.read_csv(file_path)
    data = []
    for idx in range(df.shape[0]):
        code = df.iloc[idx]['func_rename']
        label = df.iloc[idx]['vul']
        data.append((code, label))
    
    for x in data:
        code_x, _ = x
        code = ''
        for i_code in code_x.split("\n")[:-1]:
            code += i_code.strip() + ' '
        if code == '':
            print("ERROR!!!")
        
        texts.append(code.strip())


def cwe_in_fetch_codes_data_vul_non(file_path):
    labels_non = []
    codes_non = []
    labels_vul = []
    codes_vul = []

    df = pd.read_csv(file_path)
    data = []
    for idx in range(df.shape[0]):
        code = df.iloc[idx]['func_rename']
        label = df.iloc[idx]['vul']
        data.append((code, label))
    
    for x in data:
        code_x, label_x = x
        sentences = []
        for i_code in code_x.split("\n")[:-1]:
            sentences.append(i_code.strip())
        if len(sentences) == 0:
            print("ERROR!!!")
        
        if label_x == 1:
            labels_vul.append(label_x)
            codes_vul.append(sentences)
        else:
            labels_non.append(label_x)
            codes_non.append(sentences)

    return codes_non, labels_non, codes_vul, labels_vul


def cwe_in_fetch_words_data_non_vul(file_path):
    df = pd.read_csv(file_path)
    data = []
    for idx in range(df.shape[0]):
        code = df.iloc[idx]['func_rename']
        label = df.iloc[idx]['vul']
        data.append((code, label))
    
    for x in data:
        code_x, _ = x
        code = ''
        for i_code in code_x.split("\n")[:-1]:
            code += i_code.strip() + ' '
        if code == '':
            print("ERROR!!!")

        texts.append(code.strip())


def create_data_set_nd(MAX_SENT_S, MAX_SENT_LENGTH, cweid_in, cweid_out):
    """
    Create the data set as numpy arrays.
    """
    cwe_in_data_path = "data/" + cweid_in + ".csv"
    cwe_ood_data_path = "data/" + cweid_out + ".csv"

    print('Constructing data set...')
    cwe_in_data_non, cwe_in_labels_non, cwe_in_data_vul, cwe_in_labels_vul = cwe_in_fetch_codes_data_vul_non(cwe_in_data_path)
    cwe_ood_data_non, cwe_ood_labels_non, cwe_ood_data_vul, cwe_ood_labels_vul = cwe_ood_fetch_codes_data_vul_non(cwe_ood_data_path)

    cwe_in_fetch_words_data_non_vul(cwe_in_data_path)
    cwe_ood_fetch_words_data_non_vul(cwe_ood_data_path)

    words = ''
    for idx, i_text in enumerate(texts):
        if idx == len(texts) - 1:
            words += i_text
        else:
            words += i_text + ' '

    words = words.lower().split()

    vocabulary = [(" ", None)] + Counter(words).most_common(MAX_NUM_WORDS)
    vocabulary_size = len(vocabulary)

    vocabulary = np.array([word for word, _ in vocabulary])
    dictionary = {word: code for code, word in enumerate(vocabulary)}
    code_word = {code: word for code, word in enumerate(vocabulary)}

    print('Tokenizing...')
    cwe_in_data_non_wi = np.zeros((len(cwe_in_data_non), MAX_SENT_S, MAX_SENT_LENGTH), dtype='int32')
    cwe_in_data_non_wi_len = np.zeros((len(cwe_in_data_non), MAX_SENT_S, MAX_SENT_LENGTH), dtype='int32')
    cwe_in_data_vul_wi = np.zeros((len(cwe_in_data_vul), MAX_SENT_S, MAX_SENT_LENGTH), dtype='int32')
    cwe_in_data_vul_wi_len = np.zeros((len(cwe_in_data_vul), MAX_SENT_S, MAX_SENT_LENGTH), dtype='int32')

    for i, sentences in enumerate(cwe_in_data_non):
        for j, sent in enumerate(sentences):
            if j < MAX_SENT_S:
                word_tokens = sent.lower().split()
                k = 0
                for _, word in enumerate(word_tokens):
                    if k < MAX_SENT_LENGTH and dictionary[word] <= MAX_NUM_WORDS:
                        cwe_in_data_non_wi[i, j, k] = dictionary[word]
                        cwe_in_data_non_wi_len[i, j, k] = 1
                        k = k + 1

    for i, sentences in enumerate(cwe_in_data_vul):
        for j, sent in enumerate(sentences):
            if j < MAX_SENT_S:
                word_tokens = sent.lower().split()
                k = 0
                for _, word in enumerate(word_tokens):
                    if k < MAX_SENT_LENGTH and dictionary[word] <= MAX_NUM_WORDS:
                        cwe_in_data_vul_wi[i, j, k] = dictionary[word]
                        cwe_in_data_vul_wi_len[i, j, k] = 1
                        k = k + 1

    print('Tokenizing...')
    cwe_ood_data_non_wi = np.zeros((len(cwe_ood_data_non), MAX_SENT_S, MAX_SENT_LENGTH), dtype='int32')
    cwe_ood_data_non_wi_len = np.zeros((len(cwe_ood_data_non), MAX_SENT_S, MAX_SENT_LENGTH), dtype='int32')
    cwe_ood_data_vul_wi = np.zeros((len(cwe_ood_data_vul), MAX_SENT_S, MAX_SENT_LENGTH), dtype='int32')
    cwe_ood_data_vul_wi_len = np.zeros((len(cwe_ood_data_vul), MAX_SENT_S, MAX_SENT_LENGTH), dtype='int32')

    for i, sentences in enumerate(cwe_ood_data_non):
        for j, sent in enumerate(sentences):
            if j < MAX_SENT_S:
                word_tokens = sent.lower().split()
                k = 0
                for _, word in enumerate(word_tokens):
                    if k < MAX_SENT_LENGTH and dictionary[word] <= MAX_NUM_WORDS:
                        cwe_ood_data_non_wi[i, j, k] = dictionary[word]
                        cwe_ood_data_non_wi_len[i, j, k] = 1
                        k = k + 1

    for i, sentences in enumerate(cwe_ood_data_vul):
        for j, sent in enumerate(sentences):
            if j < MAX_SENT_S:
                word_tokens = sent.lower().split()
                k = 0
                for _, word in enumerate(word_tokens):
                    if k < MAX_SENT_LENGTH and dictionary[word] <= MAX_NUM_WORDS:
                        cwe_ood_data_vul_wi[i, j, k] = dictionary[word]
                        cwe_ood_data_vul_wi_len[i, j, k] = 1
                        k = k + 1

    word_index = dictionary
    index_word = code_word

    cwe_in_labels_non = np.array(cwe_in_labels_non)
    cwe_in_labels_vul = np.array(cwe_in_labels_vul)
    cwe_ood_labels_non = np.array(cwe_ood_labels_non)
    cwe_ood_labels_vul = np.array(cwe_ood_labels_vul)

    data = {
        'cwe_in_data_non_wi': cwe_in_data_non_wi,
        'cwe_in_data_vul_wi': cwe_in_data_vul_wi,
        'cwe_ood_data_non_wi': cwe_ood_data_non_wi,
        'cwe_ood_data_vul_wi': cwe_ood_data_vul_wi,
        'cwe_in_labels_non': cwe_in_labels_non,
        'cwe_in_labels_vul': cwe_in_labels_vul,
        'cwe_ood_labels_non': cwe_ood_labels_non,
        'cwe_ood_labels_vul': cwe_ood_labels_vul,
        'cwe_ood_data_non_wi_len': cwe_ood_data_non_wi_len,
        'cwe_ood_data_vul_wi_len': cwe_ood_data_vul_wi_len,
        'cwe_in_data_non_wi_len': cwe_in_data_non_wi_len,
        'cwe_in_data_vul_wi_len': cwe_in_data_vul_wi_len,
        'vocabulary_size': vocabulary_size,
        'index_word': index_word
    }

    return data
