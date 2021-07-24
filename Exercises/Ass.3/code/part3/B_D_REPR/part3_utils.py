import os
import collections
import numpy as np
import torch
import copy
import matplotlib.pyplot as plt


RAREWORDS = 200
UNKNOWN_STR = "uuunkkk"


def sentence_to_triagrams_samples(max_sentence_len, words, tags, prefix, suffix, words_len):
    sentence_len = words_len
    prefix = prefix + ["end"] * (max_sentence_len - len(words))
    suffix = suffix + ["end"] * (max_sentence_len - len(words))
    triagrams = []
    pre_window = prefix
    suf_window = suffix
    label = tags + ["end"] * (max_sentence_len - len(words))
    words = words + ["end"] * (max_sentence_len - len(words))
    window = words
    triagrams.append((window, label, pre_window, suf_window, sentence_len))
    return triagrams

def get_test_data(file_path, longest_sentence):
    split_it = " "
    longest_sentence_2 = 0
    with open(file_path, mode="r", encoding="utf-8") as f:
        words_2 = []
        flag_2 = True
        for l in f:
            if (split_it not in l) and flag_2:
                split_it = "\t"
            flag_2 = False
            if len(l) > 1:
                words_2.append(l[:-1])
            elif len(l) == 1:
                if len(words_2) > longest_sentence_2:
                    longest_sentence_2 = len(words_2)
                words_2 = []
    if longest_sentence_2 > longest_sentence:
        longest_sentence = longest_sentence_2
    test_triagram = []
    flag = True
    with open(file_path, mode="r", encoding="utf-8") as f:
        line = 0
        words, tags, prefix, suffix = [], [], [], []
        for l in f:
            if (split_it not in l) and flag:
                split_it = "\t"
            flag = False
            if len(l) > 1:
                words.append(l[:-1].lower())
                tags.append([None])
                if len(l.split(split_it)[0]) > 2:
                    prefix.append(l.split(split_it)[0][:3].lower())
                    suffix.append(l.split(split_it)[0][-3:].lower())
                else:
                    prefix.append(UNKNOWN_STR.lower())
                    suffix.append(UNKNOWN_STR.lower())
                line += 1
            elif len(l) == 1:
                if len(words) < longest_sentence + 1:
                    triagrams = sentence_to_triagrams_samples(longest_sentence, words, tags, prefix, suffix, len(words))
                else:
                    triagrams = sentence_to_triagrams_samples(longest_sentence, words[:longest_sentence], tags[:longest_sentence], prefix[:longest_sentence], suffix[:longest_sentence], longest_sentence)
                test_triagram += triagrams
                words, tags, prefix, suffix = [], [], [], []
            else:
                assert ("strange line: {} what's it?".format(l))
    return test_triagram, longest_sentence

def get_data(train_path, dev_path, test_path):
    train_triagram, dev_triagram, test_triagram = [], [], []
    split_it = " "
    with open(train_path, mode="r", encoding="utf-8") as f:
        words = []
        longest_sentence = 0
        flag = True
        for l in f:
            if (split_it not in l) and flag:
                split_it = "\t"
            flag = False
            if len(l) > 2:
                words.append(l.split(split_it)[0])
            elif len(l) == 1:
                if len(words) > longest_sentence:
                    longest_sentence = len(words)
                words = []
    with open(train_path, mode="r", encoding="utf-8") as f:
        line = 0
        words, tags, prefix, suffix = [], [], [], []
        for l in f:
            if len(l) > 2:
                words.append(l.split(split_it)[0].lower())
                tags.append(l.split(split_it)[1][:-1])
                if len(l.split(split_it)[0]) > 2:
                    prefix.append(l.split(split_it)[0][:3].lower())
                    suffix.append(l.split(split_it)[0][-3:].lower())
                else:
                    prefix.append(UNKNOWN_STR.lower())
                    suffix.append(UNKNOWN_STR.lower())
                line += 1
            elif len(l) == 1:
                triagrams = sentence_to_triagrams_samples(longest_sentence, words, tags, prefix, suffix, len(words))
                train_triagram += triagrams
                words, tags, prefix, suffix = [], [], [], []
            else:
                assert("strange line: {} what's it?".format(l))

    if dev_path is not None:
        with open(dev_path, mode="r", encoding="utf-8") as f:
            line = 0
            words, tags, prefix, suffix = [], [], [], []
            for l in f:
                if len(l) > 2:
                    words.append(l.split(split_it)[0].lower())
                    tags.append(l.split(split_it)[1][:-1])
                    if len(l.split(split_it)[0]) > 2:
                        prefix.append(l.split(split_it)[0][:3].lower())
                        suffix.append(l.split(split_it)[0][-3:].lower())
                    else:
                        prefix.append(UNKNOWN_STR.lower())
                        suffix.append(UNKNOWN_STR.lower())
                    line += 1
                elif len(l) == 1:
                    triagrams = sentence_to_triagrams_samples(longest_sentence, words, tags, prefix, suffix, len(words))
                    dev_triagram += triagrams
                    words, tags, prefix, suffix = [], [], [], []
                else:
                    assert("strange line: {} what's it?".format(l))

    if test_path is not None:
        with open(test_path, mode="r", encoding="utf-8") as f:
            line = 0
            words, tags, prefix, suffix = [], [], [], []
            for l in f:
                if len(l) > 1:
                    words.append(l[:-1].lower())
                    tags.append([None])
                    if len(l.split(split_it)[0]) > 2:
                        prefix.append(l.split(split_it)[0][:3].lower())
                        suffix.append(l.split(split_it)[0][-3:].lower())
                    else:
                        prefix.append(UNKNOWN_STR.lower())
                        suffix.append(UNKNOWN_STR.lower())
                    line += 1
                elif len(l) == 1:
                    triagrams = sentence_to_triagrams_samples(longest_sentence, words, tags, prefix, suffix, len(words))
                    test_triagram += triagrams
                    words, tags, prefix, suffix = [], [], [], []
                else:
                    assert("strange line: {} what's it?".format(l))

    return longest_sentence, train_triagram, dev_triagram, test_triagram

def prepare_data(train_data, dev_data, device, longest_sentence):
    if dev_data == []:
        dev_data = train_data[:100]
    char_train_lens = []
    char_dev_lens = []
    train_lens = [i[4] for i in train_data]
    dev_lens = [i[4] for i in dev_data]
    vocab_arr = [j for i in train_data for j in i[0]]
    vocab = set(vocab_arr)
    prefix_arr = [j for i in train_data for j in i[2]]
    suffix_arr = [j for i in train_data for j in i[3]]
    vocab_pre = set(prefix_arr)
    vocab_suf = set(suffix_arr)
    vocab_pre.add(UNKNOWN_STR)
    vocab_suf.add(UNKNOWN_STR)
    max_word_length = np.max([len(i) for i in vocab_arr])
    vocab_chars = [j for i in vocab for j in i]
    vocab_chars_set = set(vocab_chars)
    vocab_hist_char = dict(collections.Counter(vocab_chars).most_common(len(vocab_chars_set) - 5))
    vocab_new_char = set(list(vocab_hist_char.keys()))
    vocab_new_char.add("end")
    vocab_new_char.add(UNKNOWN_STR)
    char_to_ix = {character: i for i, character in enumerate(vocab_new_char)}
    vocab = set(vocab_arr)
    vocab_hist = dict(collections.Counter(vocab_arr).most_common(len(vocab) - RAREWORDS))
    vocab_new = set(list(vocab_hist.keys()))
    vocab_to_learn = vocab_new.copy()
    vocab_to_learn.add(UNKNOWN_STR)
    labels_vocab = set([k for i in train_data for k in i[1]])
    word_to_ix = {word: i for i, word in enumerate(vocab_to_learn)}
    label_to_ix = {label: i for i, label in enumerate(labels_vocab)}
    pre_to_ix = {word: i for i, word in enumerate(vocab_pre)}
    suf_to_ix = {word: i for i, word in enumerate(vocab_suf)}
    x = [i[0] for i in train_data]
    y = [i[1] for i in train_data]
    pre_x = [i[2] for i in train_data]
    suf_x = [i[3] for i in train_data]
    char_x = copy.deepcopy(x)
    for i in range(len(x)):
        y[i] = [label_to_ix[k] for k in y[i]]
        for j in range(longest_sentence):
            idx = []
            for k in range(len(char_x[i][j])):
                if char_x[i][j][k] in char_to_ix.keys():
                    idx.append(char_to_ix[char_x[i][j][k]])
                else:
                    idx.append(char_to_ix[UNKNOWN_STR])
            char_train_lens.append(len(idx))
            idx += [char_to_ix["end"]] * (max_word_length - len(idx))
            char_x[i][j] = idx
            if x[i][j] in vocab_new:
                x[i][j] = word_to_ix[x[i][j]]
            else:
                x[i][j] = word_to_ix[UNKNOWN_STR]
            if pre_x[i][j] in vocab_pre:
                pre_x[i][j] = pre_to_ix[pre_x[i][j]]
            else:
                pre_x[i][j] = pre_to_ix[UNKNOWN_STR]
            if suf_x[i][j] in vocab_suf:
                suf_x[i][j] = suf_to_ix[suf_x[i][j]]
            else:
                suf_x[i][j] = suf_to_ix[UNKNOWN_STR]
    dev_x = [i[0] for i in dev_data]
    dev_y = [i[1] for i in dev_data]
    pre_dev = [i[2] for i in dev_data]
    suf_dev = [i[3] for i in dev_data]
    char_dev = copy.deepcopy(dev_x)
    for i in range(len(dev_x)):
        dev_y[i] = [label_to_ix[k] for k in dev_y[i]]
        for j in range(longest_sentence):
            idx = []
            for k in range(len(char_dev[i][j])):
                if char_dev[i][j][k] in char_to_ix.keys():
                    idx.append(char_to_ix[char_dev[i][j][k]])
                else:
                    idx.append(char_to_ix[UNKNOWN_STR])
            char_dev_lens.append(len(idx))
            idx += [char_to_ix["end"]] * (max_word_length - len(idx))
            char_dev[i][j] = idx
            if dev_x[i][j] in vocab_new:
                dev_x[i][j] = word_to_ix[dev_x[i][j]]
            else:
                dev_x[i][j] = word_to_ix[UNKNOWN_STR]
            if pre_dev[i][j] in vocab_pre:
                pre_dev[i][j] = pre_to_ix[pre_dev[i][j]]
            else:
                pre_dev[i][j] = pre_to_ix[UNKNOWN_STR]
            if suf_dev[i][j] in vocab_suf:
                suf_dev[i][j] = suf_to_ix[suf_dev[i][j]]
            else:
                suf_dev[i][j] = suf_to_ix[UNKNOWN_STR]
    x_all = torch.LongTensor(x).to(torch.device(device.type))
    y_all = torch.LongTensor(y).to(torch.device(device.type))
    x_dev = torch.LongTensor(dev_x).to(torch.device(device.type))
    y_dev = torch.LongTensor(dev_y).to(torch.device(device.type))
    suff_dev = torch.LongTensor(suf_dev).to(torch.device(device.type))
    pref_dev = torch.LongTensor(pre_dev).to(torch.device(device.type))
    pre_x_all = torch.LongTensor(pre_x).to(torch.device(device.type))
    suf_x_all = torch.LongTensor(suf_x).to(torch.device(device.type))
    char_x = torch.LongTensor(char_x).to(torch.device(device.type))
    char_dev = torch.LongTensor(char_dev).to(torch.device(device.type))
    return char_train_lens, char_dev_lens, train_lens, dev_lens, vocab_to_learn, vocab_pre, vocab_suf, labels_vocab, vocab_chars, x_all, y_all, x_dev, y_dev, suff_dev, pref_dev, pre_x_all, suf_x_all, word_to_ix, label_to_ix, pre_to_ix, suf_to_ix, char_to_ix, max_word_length, char_x, char_dev

def prepare_test_data(test_data, word_to_ix, label_to_ix, char_to_ix, pre_to_ix, suf_to_ix, longest_sentence, max_word_length, device):
    dev_x = [i[0] for i in test_data]
    pre_dev = [i[2] for i in test_data]
    suf_dev = [i[3] for i in test_data]
    dev_lens = [i[4] for i in test_data]
    char_dev_lens = []
    char_dev = copy.deepcopy(dev_x)
    for i in range(len(dev_x)):
        for j in range(longest_sentence):
            idx = []
            for k in range(len(char_dev[i][j])):
                if char_dev[i][j][k] in char_to_ix.keys():
                    idx.append(char_to_ix[char_dev[i][j][k]])
                else:
                    idx.append(char_to_ix[UNKNOWN_STR])
            char_dev_lens.append(len(idx))
            idx += [char_to_ix["end"]] * (max_word_length - len(idx))
            char_dev[i][j] = idx
            if dev_x[i][j] in word_to_ix.keys():
                dev_x[i][j] = word_to_ix[dev_x[i][j]]
            else:
                dev_x[i][j] = word_to_ix[UNKNOWN_STR]
            if pre_dev[i][j] in pre_to_ix.keys():
                pre_dev[i][j] = pre_to_ix[pre_dev[i][j]]
            else:
                pre_dev[i][j] = pre_to_ix[UNKNOWN_STR]
            if suf_dev[i][j] in suf_to_ix.keys():
                suf_dev[i][j] = suf_to_ix[suf_dev[i][j]]
            else:
                suf_dev[i][j] = suf_to_ix[UNKNOWN_STR]
    labels = [k for k in label_to_ix.keys()]
    x_dev = torch.LongTensor(dev_x).to(torch.device(device.type))
    suff_dev = torch.LongTensor(suf_dev).to(torch.device(device.type))
    pref_dev = torch.LongTensor(pre_dev).to(torch.device(device.type))
    char_dev = torch.LongTensor(char_dev).to(torch.device(device.type))
    return char_dev_lens, x_dev, pref_dev, suff_dev, char_dev, labels, dev_lens


def saving_outputs(losses, train_acc, dev_acc, dev_loss, name_for_outputs):
    plt.figure(1)
    plt.plot(losses)  # ([5 * (i + 1) for i in range(len(losses))], losses)
    plt.title("Loss on training")
    plt.savefig("Training Loss {}.jpg".format(name_for_outputs))
    plt.close()
    plt.figure(2)
    plt.plot(train_acc)
    plt.title("Acc on training")
    plt.savefig("Training Acc {}.jpg".format(name_for_outputs))
    plt.close()
    plt.figure(3)
    plt.plot(dev_acc)
    plt.title("Acc on dev")
    plt.savefig("Dev Acc {}.jpg".format(name_for_outputs))
    plt.close()
    plt.figure(4)
    plt.plot(dev_loss)
    plt.title("Loss on dev")
    plt.savefig("Dev Loss {}.jpg".format(name_for_outputs))
    plt.close()
