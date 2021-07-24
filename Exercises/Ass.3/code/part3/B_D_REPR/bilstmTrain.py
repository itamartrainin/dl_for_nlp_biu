import torch
from torch import nn
import torch.optim as optim
from operator import itemgetter
from part3_utils import *
import sys


EMBEDDING_LENGTH = 10
CHAR_EMBEDDING_LENGTH = 10
BATCH_SIZE = 1
SAVE_OUTPUT = True


class c(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, prefix_size, suffix_size, tagging_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, EMBEDDING_LENGTH)
        self.embeddings_pre = nn.Embedding(prefix_size, EMBEDDING_LENGTH)
        self.embeddings_suf = nn.Embedding(suffix_size, EMBEDDING_LENGTH)
        self.cell = nn.LSTM(EMBEDDING_LENGTH, hidden_size, 2, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, tagging_size)
        self.device_to = device

    def forward(self, inputs, input_lens):
        embeds_pre = self.embeddings_pre(inputs[0].T)
        embeds = self.embeddings(inputs[1].T)
        embeds_suf = self.embeddings_suf(inputs[2].T)
        cat_embedded = embeds_pre + embeds + embeds_suf
        cat_embedded = nn.utils.rnn.pack_padded_sequence(cat_embedded, input_lens, enforce_sorted=False)
        output, _ = self.cell(cat_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        last_layer = self.linear2(nn.functional.tanh(self.linear(output)))
        log_probs = torch.nn.functional.log_softmax(last_layer, dim=2)
        return log_probs

class a(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, tagging_size, device):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, EMBEDDING_LENGTH)
        self.cell = nn.LSTM(EMBEDDING_LENGTH, hidden_size, 2, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, tagging_size)
        self.device_to = device

    def forward(self, input, input_lens):
        embd = self.embeddings(input.T)
        embd = nn.utils.rnn.pack_padded_sequence(embd, input_lens, enforce_sorted=False)
        output, _ = self.cell(embd)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        lin_output = self.linear2(nn.functional.tanh(self.linear(output)))
        log_probs = torch.nn.functional.log_softmax(lin_output, dim=2)
        return log_probs

class d(torch.nn.Module):
    def __init__(self, vocab_size, tagging_size, char_vocab_size, hidden_size, device):
        super().__init__()
        self.embeddings_b = nn.Embedding(char_vocab_size, CHAR_EMBEDDING_LENGTH)
        self.embd_lstm_b = nn.LSTM(CHAR_EMBEDDING_LENGTH, hidden_size, 1)
        self.embeddings_a = nn.Embedding(vocab_size, EMBEDDING_LENGTH)
        self.linear_combined = nn.Linear(hidden_size + EMBEDDING_LENGTH, hidden_size)
        self.cell = nn.LSTM(hidden_size, hidden_size, 2, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.linear_final = nn.Linear(hidden_size, tagging_size)
        self.device_to = device

    def forward(self, input, input_lens, char_len):
        input[0] = input[0][:, :, :char_len[0]]
        embd = self.embeddings_b(input[0])
        embd = embd.view((embd.shape[0] * embd.shape[1], embd.shape[2], embd.shape[3]))
        embd = embd.permute((1, 0, 2))
        output, _ = self.embd_lstm_b(embd)
        output = output[-1, :, :]
        output = output.view((input[0].shape[0], input[0].shape[1], output.shape[1]))
        output_b = output.permute((1, 0, 2))
        embd_a = self.embeddings_a(input[1].T)
        lin_combined = nn.functional.tanh(self.linear_combined(torch.cat((output_b, embd_a), dim=2)))
        lin_combined = nn.utils.rnn.pack_padded_sequence(lin_combined, input_lens, enforce_sorted=False)
        output2, _ = self.cell(lin_combined)
        output2, _ = nn.utils.rnn.pad_packed_sequence(output2)
        last_layer = self.linear_final(nn.functional.tanh(self.linear(output2)))
        log_probs = torch.nn.functional.log_softmax(last_layer, dim=2)
        return log_probs


class b(torch.nn.Module):
    def __init__(self, tagging_size, char_vocab_size, hidden_size, device):
        super().__init__()
        self.embeddings = nn.Embedding(char_vocab_size, CHAR_EMBEDDING_LENGTH)
        self.embd_lstm = nn.LSTM(CHAR_EMBEDDING_LENGTH, hidden_size, 1)
        self.cell = nn.LSTM(hidden_size, hidden_size, 2, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, hidden_size)
        self.linear_final = nn.Linear(hidden_size, tagging_size)
        self.device_to = device

    def forward(self, input, input_lens, char_len):
        input = input[:, :, :char_len[0]]
        embd = self.embeddings(input)
        embd = embd.view((embd.shape[0] * embd.shape[1], embd.shape[2], embd.shape[3]))
        embd = embd.permute((1, 0, 2))
        output, _ = self.embd_lstm(embd)
        output = output[-1, :, :]
        output = output.view((input.shape[0], input.shape[1], output.shape[1]))
        output = output.permute((1, 0, 2))
        output = nn.utils.rnn.pack_padded_sequence(output, input_lens, enforce_sorted=False)
        output2, _ = self.cell(output)
        output2, _ = nn.utils.rnn.pad_packed_sequence(output2)
        last_layer = self.linear_final(nn.functional.tanh(self.linear(output2)))
        log_probs = torch.nn.functional.log_softmax(last_layer, dim=2)
        return log_probs

def learning_tagger(longest_sentence, assignment, train_data, dev_data, device=torch.device("cpu"), num_iterations=10, name_for_outputs="NER", hidden_size=150, lr=1e-1, flag_ner=False):
    char_train_lens, char_dev_lens, train_lens, dev_lens, vocab_to_learn, vocab_pre, vocab_suf, labels_vocab, vocab_chars, x_all, y_all, x_dev, y_dev, suff_dev, pref_dev, pre_x_all, suf_x_all, word_to_ix, label_to_ix, pre_to_ix, suf_to_ix, char_to_ix, max_word_length, char_x, char_dev = prepare_data(train_data, dev_data, device, longest_sentence)
    if assignment == "c":
        our_model = c(len(vocab_to_learn), hidden_size, len(vocab_pre), len(vocab_suf), len(labels_vocab))
    elif assignment == "a":
        our_model = a(len(vocab_to_learn), hidden_size, len(labels_vocab), device)
    elif assignment == "b":
        our_model = b(len(labels_vocab), len(vocab_chars), hidden_size, device)
    else:
        our_model = d(len(vocab_to_learn), len(labels_vocab), len(vocab_chars), hidden_size, device)
    our_model.to(torch.device(device))
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(our_model.parameters(), lr=lr)
    losses = []
    train_acc = []
    dev_acc = []
    dev_loss = []
    for epoch in range(num_iterations):
        total_loss = 0
        total_acc = 0
        total_examples = 0
        counter = 0
        for batch in range(0, len(train_data), BATCH_SIZE):
            if batch < len(train_data):
                window_idxs = x_all[batch:batch + BATCH_SIZE]
                label_idxs = y_all[batch:batch + BATCH_SIZE]
                prefix_idxs = pre_x_all[batch:batch + BATCH_SIZE]
                suffix_idxs = suf_x_all[batch:batch + BATCH_SIZE]
                char_idxs = char_x[batch:batch + BATCH_SIZE]
                lens_idx = train_lens[batch:batch + BATCH_SIZE]
                char_lens_idxs = char_train_lens[batch:batch + BATCH_SIZE]
                num_samples = BATCH_SIZE
            counter += num_samples
            our_model.zero_grad()
            if assignment == "c":
                log_probs = our_model.forward([prefix_idxs, window_idxs, suffix_idxs], lens_idx)
            elif assignment == "b":
                log_probs = our_model.forward(char_idxs, lens_idx, char_lens_idxs)
            elif assignment == "a":
                log_probs = our_model.forward(window_idxs, lens_idx)
            else:
                log_probs = our_model.forward([char_idxs, window_idxs], lens_idx, char_lens_idxs)
            y_hat = torch.argmax(log_probs, dim=2)
            loss, acc, num_samples = removing_end_calculate_acc_loss(loss_func, log_probs, torch.enable_grad, y_hat, label_idxs, label_to_ix, flag_ner)    # loss_func(log_probs.permute(1, 2, 0), label_idxs)
            if num_samples > 0:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_acc += acc
                total_examples += num_samples
            with torch.no_grad():
                if counter > 499:
                    acc_dev_total = 0
                    num_dev_samples_total = 0
                    loss_total_dev = 0
                    for k in range(x_dev.shape[0]):
                        if assignment == "c":
                            log_probs_dev = our_model.forward([pref_dev[k:k+1], x_dev[k:k+1], suff_dev[k:k+1]], dev_lens[k:k+1])
                        elif assignment == "b":
                            log_probs_dev = our_model.forward(char_dev[k:k+1], dev_lens[k:k+1], char_dev_lens[k:k+1])
                        elif assignment == "a":
                            log_probs_dev = our_model.forward(x_dev[k:k+1], dev_lens[k:k+1])
                        else:
                            log_probs_dev = our_model.forward([char_dev[k:k+1], x_dev[k:k+1]], dev_lens[k:k+1], char_dev_lens[k:k+1])
                        y_hat_dev = torch.argmax(log_probs_dev, dim=2)
                        loss_dev, acc_dev, num_dev_samples = removing_end_calculate_acc_loss(loss_func, log_probs_dev, torch.no_grad, y_hat_dev, y_dev[k:k+1], label_to_ix, flag_ner)
                        acc_dev_total += acc_dev
                        num_dev_samples_total += num_dev_samples
                        loss_total_dev += loss_dev
                    dev_loss.append(loss_total_dev)
                    dev_acc.append(acc_dev_total * 100 / num_dev_samples_total)
                    print("Temporary acc on dev: {}".format(dev_acc[-1]))
                    del y_hat_dev
                    counter = 0
        with torch.no_grad():
            losses.append(total_loss)
            train_acc.append((total_acc * 100) / total_examples)
            print("epoch: {} train acc: {} train loss: {} dev acc: {} dev loss: {}".format(epoch + 1, train_acc[-1], losses[-1], dev_acc[-1], dev_loss[-1]))

    if SAVE_OUTPUT:
        saving_outputs(losses, train_acc, dev_acc, dev_loss, name_for_outputs)
    return our_model, label_to_ix, word_to_ix, char_to_ix, pre_to_ix, suf_to_ix, max_word_length, dev_acc, dev_loss

def removing_end_calculate_acc_loss(loss_func, log_probs, grad_flag, y_hat, true_labels, label_to_ix, flag_ner=False):
    with grad_flag():
        y_hat = y_hat.permute((1, 0))   # batch x max_sentence_size_in_batch
        true_labels = true_labels[:, :y_hat.shape[1]]   # batch x batch x max_sentence_size_in_batch
        y_hat = y_hat.reshape((y_hat.shape[0] * y_hat.shape[1], 1))
        true_labels = true_labels.reshape(((y_hat.shape[0] * y_hat.shape[1], 1)))
        if flag_ner:
            O_O = (label_to_ix["O"] != true_labels) + (true_labels != y_hat)
            mask = (label_to_ix["end"] != true_labels) & O_O
        else:
            mask = (label_to_ix["end"] != true_labels)
        num_examples = mask.float().sum()
        if num_examples > 0:
            y_hat_new = y_hat[mask]
            true_labels_new = true_labels[mask]
            log_probs = log_probs.permute((1, 0, 2))
            log_probs = log_probs.reshape((log_probs.shape[0] * log_probs.shape[1], log_probs.shape[2]))
            masking_2d = torch.broadcast_tensors(log_probs, mask)[1]
            log_probs = log_probs[masking_2d]
            log_probs = log_probs.reshape((y_hat_new.shape[0], len(label_to_ix)))
            loss = loss_func(log_probs, true_labels_new)
            acc = (y_hat_new == true_labels_new).float().sum()
            return loss, acc, num_examples
        else:
            return 0, 0, 0

def test_our_model(longest_sentence, assignment, model, test_data_path, label_to_ix, max_word_length=50, word_to_ix={}, pre_to_ix={}, suf_to_ix={}, char_to_ix={}, device=torch.device("cpu"), name_for_outputs="test_model"):
    test_data, longest_sentence = get_test_data(test_data_path, longest_sentence)
    char_dev_lens, x_dev, pref_dev, suff_dev, char_dev, labels, dev_lens = prepare_test_data(test_data, word_to_ix, label_to_ix, char_to_ix, pre_to_ix, suf_to_ix, longest_sentence, max_word_length, device)
    labels_hat = []
    for k in range(x_dev.shape[0]):
        if assignment == "c":
            log_probs_dev = model.forward([pref_dev[k:k+1], x_dev[k:k+1], suff_dev[k:k+1]], dev_lens[k:k+1])
        elif assignment == "b":
            log_probs_dev = model.forward(char_dev[k:k+1], dev_lens[k:k+1], char_dev_lens[k:k+1])
        elif assignment == "a":
            log_probs_dev = model.forward(x_dev[k:k+1], dev_lens[k:k+1])
        else:
            log_probs_dev = model.forward([char_dev[k:k+1], x_dev[k:k+1]], dev_lens[k:k+1], char_dev_lens[k:k+1])
        y_hat_dev = torch.argmax(log_probs_dev, dim=2).T
        y_hat = y_hat_dev.reshape((y_hat_dev.shape[0] * y_hat_dev.shape[1], 1))
        y_hat = y_hat[:dev_lens[k:k+1][0]]
        labels_hat += list(itemgetter(*y_hat)(labels))
    if SAVE_OUTPUT:
        with open("Part3.{}".format(name_for_outputs), mode="w", encoding="utf-8") as f:
            glob_counter = 0
            for len_sen in dev_lens:
                counter = 0
                while counter < len_sen:
                    f.write(labels_hat[glob_counter] + "\n")
                    counter += 1
                    glob_counter += 1
                f.write("\n")

if __name__=="__main__":
    device = torch.device("cuda")
    if len(sys.argv) == 4:
        assignment, train_path, model_path = sys.argv[1:]
        dev_path, test_path, name_for_outputs, flag_ner = None, None, "bilstm", False
    elif len(sys.argv) == 6:
        assignment, train_path, model_path, dev_path, test_path = sys.argv[1:]
        name_for_outputs, flag_ner = "bilstm", False
    elif len(sys.argv) == 7:
        assignment, train_path, model_path, dev_path, test_path, name_for_outputs = sys.argv[1:]
        flag_ner = False
    elif len(sys.argv) == 8:
        assignment, train_path, model_path, dev_path, test_path, name_for_outputs, flag_ner = sys.argv[1:]
    longest_sentence, train, dev, test = get_data(train_path, dev_path, test_path)
    our_model, label_to_ix, word_to_ix, char_to_ix, pre_to_ix, suf_to_ix, max_word_length, dev_acc, dev_loss = learning_tagger(longest_sentence, assignment, train, dev, device, 20, name_for_outputs, 20, 1e-1, flag_ner)
    torch.save([longest_sentence, assignment, our_model, label_to_ix, max_word_length, word_to_ix, pre_to_ix, suf_to_ix, char_to_ix, device, name_for_outputs], model_path)
    if test_path is not None:
        test_our_model(longest_sentence, assignment, our_model, test_path, label_to_ix, max_word_length, word_to_ix, pre_to_ix, suf_to_ix, char_to_ix, device, name_for_outputs)
    torch.save([dev_acc, dev_loss], "dev_acc_loss_" + name_for_outputs)
