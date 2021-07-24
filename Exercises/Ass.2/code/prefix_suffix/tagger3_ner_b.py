import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

STUDENT = {'name': 'Itamar Trainin', 'ID': '315425967'}


def read_data(base_dir, task, fname, window_size, freq_bound, balance_coef, psfix_size):
    data = []

    seen_words = set()
    seen_labels = set()

    vocab = set()
    labels = set()
    psfix = set()

    word_freq = {}
    label_freq = {}

    window_delta = int(window_size/2)

    #  read all words and labels from file and count their frequencies
    file = open(base_dir + '/' + task + '/' + fname, 'r', encoding='utf-8')
    for line in file.readlines():
        if line != '\n':
            if task == 'ner':
                word, label = line.lower().strip().split("\t", 1)
            else:
                word, label = line.lower().strip().split(" ", 1)

            if len(word) < 3:
                pre = '-under-'
                suf = '-under-'
            else:
                pre = word[:psfix_size]
                suf = word[-psfix_size:]

            data.append(((word, pre, suf), label))

            if word in seen_words:
                word_freq[word] += 1
            else:
                seen_words.add(word)
                word_freq[word] = 1

            if label in seen_labels:
                label_freq[label] += 1
            else:
                if label != 'o':
                    seen_labels.add(label)
                    label_freq[label] = 1

    # collect set of unfrequent words
    unfrequent_words = []
    for word in word_freq:
        if word_freq[word] < freq_bound:
            unfrequent_words.append(word)

    # take only at most 100 unfrequent words
    if len(unfrequent_words) > 100:
        unfrequent_words = unfrequent_words[:100]
    else:
        print('Less than 100 different unfrequent words where found!!')

    # find ave of frequencies of labels appearances
    labels_freq_max = max(list(label_freq.values()))
    balanced_o_freq = int(balance_coef * labels_freq_max)
    word_count = sum(list(label_freq.values())) + balanced_o_freq

    # replace unfrequent words with '-unknown-' and generate vocab and labels
    for i in range(len(data)):
        token, label = data[i]
        word, pre, suf = token
        # Replace unknown words with -unknown- token
        if word in unfrequent_words:
            data[i] = (('-unknown-', '-unknown-'[:psfix_size], '-unknown-'[-psfix_size:]), 'o')
            vocab.add(data[i][0][0])
            psfix.add(data[i][0][1])
            psfix.add(data[i][0][2])
            labels.add('o')
        else:
            vocab.add(word)
            psfix.add(pre)
            psfix.add(suf)
            labels.add(label)

    # add -start- and -end- tokens at begingin and end to allow window operations
    for i in range(window_delta):
        if(task == 'ner'):
            data.insert(0, (('-ped-', '-ped-'[:psfix_size], '-ped-'[-psfix_size:]), 'o'))
            data.append((('-ped-', '-ped-'[:psfix_size], '-ped-'[-psfix_size:]), 'o'))
            vocab.add('-ped-')
            psfix.add('-ped-'[:psfix_size])
            psfix.add('-ped-'[-psfix_size:])
            labels.add('o')
        else:
            data.insert(0, (('\'\'', '-under-', '-under-'), '\'\''))
            data.append((('\'\'', '-under-', '-under-'), '\'\''))
            vocab.add('\'\'')
            psfix.add('-under-')
            labels.add('\'\'')

    word_ix = {}
    for word in vocab:
        word_ix[word] = len(word_ix)

    label_ix = {}
    for label in labels:
        label_ix[label] = len(label_ix)

    psfix_ix = {}
    for ps in psfix:
        psfix_ix[ps] = len(psfix_ix)

    file.close()

    return data, word_freq, word_ix, label_ix, psfix_ix, balanced_o_freq, word_count


def generate_X_y(task, part, data, E, window_size, word_to_ix, data_word_to_ix, label_to_ix, psfix_to_ix, o_bound, word_count, psfix_size):
    window_delta = int(window_size/2)
    words_in_data = len(data) - window_delta * 2

    X = []
    X_pre = []
    X_suf = []
    y = []

    o_counter = 0
    for idx in range(words_in_data):
        # Ignore over windows labeled 'o'
        center_label = data[idx + window_delta][1]
        if task == 'ner' and center_label == 'o':
            if o_counter < o_bound:
                o_counter += 1
            else:
                continue

        X_i = []
        X_pre_i = []
        X_suf_i = []
        for i in range(window_size):
            token, label = data[idx + i]
            word, pre, suf = token
            if part == 'b':
                if word in word_to_ix:
                    word_emb = E[word_to_ix[word]]
                else:
                    word_emb = E[word_to_ix['-unknown-']]
            else:
                if word in data_word_to_ix:
                    word_emb = data_word_to_ix[word]
                else:
                    word_emb = data_word_to_ix['-unknown-']

            X_i.append(word_emb)

            if pre in psfix_to_ix:
                pre_ix = psfix_to_ix[pre]
            else:
                pre_ix = psfix_to_ix['-unknown-'[:psfix_size]]
            X_pre_i.append(pre_ix)

            if suf in psfix_to_ix:
                suf_ix = psfix_to_ix[suf]
            else:
                suf_ix = psfix_to_ix['-unknown-'[-psfix_size:]]
            X_suf_i.append(suf_ix)
        X.append(X_i)
        X_pre.append(X_pre_i)
        X_suf.append(X_suf_i)
        y.append(label_to_ix[center_label])

    X_t = torch.tensor(X)
    X_pre_t = torch.tensor(X_pre)
    X_suf_t = torch.tensor(X_suf)
    y_t = torch.tensor(y)

    if part == 'b':
        X_t = X_t.view(X_t.size(0), X_t.size(1) * X_t.size(2)).float()
    else:
        X_t = X_t.long()

    return X_t, X_pre_t.long(), X_suf_t.long(),  y_t.long()


class Network(nn.Module):
    def __init__(self, task, part, window_size, word_input_size, embd_input_size, embd_size, hidden_size, output_size, lr, batch_size, iterations):
        super(Network, self).__init__()

        self.task = task
        self.part = part

        self.lr = lr
        self.batch_size = batch_size
        self.iterations = iterations

        self.windowSize = window_size
        self.wordInputSize = word_input_size
        self.embdInputSize = embd_input_size
        self.embdSize = embd_size
        self.hiddenSize = hidden_size
        self.outputSize = output_size

        self.E = nn.Embedding(self.embdInputSize, self.embdSize)
        self.E.weight.requires_grad = True
        self.E_words = nn.Embedding(self.wordInputSize, self.embdSize)
        self.E_words.weight.requires_grad = True
        self.hidden = nn.Linear(self.embdSize * self.windowSize, self.hiddenSize)
        self.hidden.weight.requires_grad = True
        self.tanh = nn.Tanh()
        self.output = nn.Linear(self.hiddenSize, self.outputSize)
        self.output.weight.requires_grad = True
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.loss_function = nn.NLLLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)


    def forward(self, x, x_pre, x_suf):
        e_pre = self.E(x_pre)
        e_suf = self.E(x_suf)
        if self.part == 'a':
            x = self.E_words(x)
            x = x.view(x.size(0), x.size(1) * x.size(2))
        x = x + e_pre.view(e_pre.size(0), e_pre.size(1) * e_pre.size(2)) + e_suf.view(e_suf.size(0), e_suf.size(1) * e_suf.size(2))
        x = self.hidden(x)
        x = self.tanh(x)
        x = self.output(x)
        x = self.log_softmax(x)
        return x

    def accuracy_on_data(self, data):
        X, X_pre, X_suf, y = data
        good = 0.0

        log_probs = self.forward(X, X_pre, X_suf)
        pred = torch.argmax(log_probs, dim=1)
        loss = self.loss_function(log_probs, y)

        correct = (pred == y)
        good = torch.sum(correct)

        return float(good) / float(y.size(0)), loss

    def train_model(self, train, dev):
        X, X_pre, X_suf, y = train
        num_samples = X.size(0)
        num_of_batches = int(num_samples / self.batch_size) + 1

        dev_loss_arr = []
        dev_accu_arr = []

        print('-train started-')

        for epoch in range(self.iterations):
            cum_loss = 0.0  # total loss in this iteration.
            idx_s = torch.randperm(num_samples)
            X_s = X[idx_s]
            X_pre_s = X_pre[idx_s]
            X_suf_s = X_suf[idx_s]
            y_s = y[idx_s]
            for idx in range(num_of_batches):

                self.zero_grad()

                X_batch = X_s[idx * self.batch_size:(idx + 1) * self.batch_size]
                X_pre_batch = X_pre_s[idx * self.batch_size:(idx + 1) * self.batch_size]
                X_suf_batch = X_suf_s[idx * self.batch_size:(idx + 1) * self.batch_size]
                y_batch = y_s[idx * self.batch_size:(idx + 1) * self.batch_size]

                log_probs = self.forward(X_batch, X_pre_batch, X_suf_batch)

                loss = self.loss_function(log_probs, y_batch)
                cum_loss += loss
                loss.backward()
                self.optimizer.step()

            train_loss = cum_loss / num_of_batches
            train_precision, _ = self.accuracy_on_data(train)
            dev_precision, dev_loss = self.accuracy_on_data(dev)
            dev_loss_arr.append(dev_loss * 100)
            dev_accu_arr.append(dev_precision * 100)
            print('{} / {}, {} / {}, {}'.format(epoch, train_loss, train_precision, dev_loss, dev_precision))

        plt.plot(range(self.iterations), dev_loss_arr, label='Dev Loss')
        plt.plot(range(self.iterations), dev_accu_arr, label='Dev Accuracy')
        plt.xlabel('# of iterations')
        plt.ylabel('accuracy (%) / loss (x10^-2)')
        plt.xlim(0, self.iterations)
        # plt.ylim(0, 100)
        plt.title(task)
        plt.legend()
        plt.show()


if __name__ == '__main__':

    task = 'ner'
    # task = 'pos'
    # part = 'a'
    part = 'b'
    ignore = 'o'

    if task == 'ner':
        lr = 0.005
        batch_size = 64
        iterations = 50
    else:
        lr = 0.1
        batch_size = 512
        iterations = 150

    base_dir = 'C:/Users/Itamar Trainin/Documents/DL for NLP/Exercises/Ass.2/data'
    window_size = 5
    freq_bound = 2
    balance_coef = 1.3
    psfix_size = 3

    # Load embedding matrix
    E = np.loadtxt("wordVectors")

    # Load vocabulary
    vocabf = open('vocab', 'r', encoding='utf-8')
    vocab = [line.strip() for line in vocabf.readlines()]

    # Create word to index mapping
    word_to_ix = {}
    for word in vocab:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

    train, word_freq, data_word_to_ix, label_to_ix, psfix_to_ix, o_bound, word_count = read_data(base_dir, task,
                                                                                                 'train', window_size,
                                                                                                 freq_bound,
                                                                                                 balance_coef,
                                                                                                 psfix_size)
    dev, _, _, _, _, _, _ = read_data(base_dir, task, 'dev', window_size, freq_bound, balance_coef, psfix_size)

    word_to_ix['-unknown-'] = word_to_ix['UUUNKKK']

    X_train, X_pre_train, X_suf_train, y_train = generate_X_y(task, part, train, E, window_size, word_to_ix,
                                                              data_word_to_ix, label_to_ix, psfix_to_ix, o_bound,
                                                              word_count, psfix_size)
    X_dev, X_pre_dev, X_suf_dev, y_dev = generate_X_y(task, part, dev, E, window_size, word_to_ix, data_word_to_ix,
                                                      label_to_ix, psfix_to_ix, o_bound, word_count, psfix_size)

    word_input_size = len(data_word_to_ix)
    embd_input_size = len(psfix_to_ix)
    embd_size = E.shape[1]
    if task == 'ner':
        hidden_size = 300
    else:
        hidden_size = 50
    output_size = len(label_to_ix)

    model = Network(task, part, window_size, word_input_size, embd_input_size, embd_size, hidden_size, output_size, lr, batch_size, iterations)

    model.train_model([X_train, X_pre_train, X_suf_train, y_train], [X_dev, X_pre_dev, X_suf_dev, y_dev])

    # test model
    testf = open(base_dir + '/' + task + '/test', 'r', encoding='utf-8')
    test_outputf = open('test3_' + part + '.' + task, 'w', encoding='utf-8')
    test = [word.strip().lower() for word in testf.readlines()]

    window_delta = int(window_size / 2)
    for i in range(window_delta):
        if(task == 'ner'):
            test.insert(0, ('-ped-', 'o'))
            test.append(('-ped-', 'o'))
        else:
            test.insert(0, ('\'\'', '\'\''))
            test.append(('\'\'', '\'\''))

    X_test = []
    X_pre_test = []
    X_suf_test = []
    for idx in range(len(test) - window_delta * 2):
        X_i = []
        X_pre_i = []
        X_suf_i = []
        for i in range(window_size):
            word = test[idx + i]
            pre = word[:psfix_size]
            suf = word[-psfix_size:]
            if part == 'b':
                if word in word_to_ix:
                    word_emb = E[word_to_ix[word]]
                else:
                    word_emb = E[word_to_ix['-unknown-']]
            else:
                if word in data_word_to_ix:
                    word_emb = data_word_to_ix[word]
                else:
                    word_emb = data_word_to_ix['-unknown-']
            X_i.append(word_emb)

            if pre in psfix_to_ix:
                pre_ix = psfix_to_ix[pre]
            else:
                pre_ix = psfix_to_ix['-unknown-'[:psfix_size]]
            X_pre_i.append(pre_ix)

            if suf in psfix_to_ix:
                suf_ix = psfix_to_ix[suf]
            else:
                suf_ix = psfix_to_ix['-unknown-'[-psfix_size:]]
            X_suf_i.append(suf_ix)
        X_test.append(X_i)
        X_pre_test.append(X_pre_i)
        X_suf_test.append(X_suf_i)

    X_test_t = torch.tensor(X_test)
    X_pre_test_t = torch.tensor(X_pre_test).long()
    X_suf_test_t = torch.tensor(X_suf_test).long()

    if part == 'b':
        X_test_t = X_test_t.view(X_test_t.size(0), X_test_t.size(1) * X_test_t.size(2)).float()
    else:
        X_test_t = X_test_t.long()

    log_probs = model.forward(X_test_t, X_pre_test_t, X_suf_test_t)
    pred = torch.argmax(log_probs, dim=1)

    ix_to_label = {label_to_ix[label]: label for label in label_to_ix}

    for p in pred:
        test_outputf.write(str(ix_to_label[int(p)]) + '\n')

    testf.close()
    test_outputf.close()