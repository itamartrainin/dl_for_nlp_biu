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
    device = torch.device("cpu")
    assignment, model_path, test_path = sys.argv[1:]
    longest_sentence, assignment, our_model, label_to_ix, max_word_length, word_to_ix, pre_to_ix, suf_to_ix, char_to_ix, device, name_for_outputs = torch.load(model_path)
    test_our_model(longest_sentence, assignment, our_model, test_path, label_to_ix, max_word_length, word_to_ix, pre_to_ix, suf_to_ix, char_to_ix, device, "Predicted File")

