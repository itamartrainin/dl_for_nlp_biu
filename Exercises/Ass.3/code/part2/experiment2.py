from torch import nn, autograd
import torch
import matplotlib.pyplot as plt

BATCH_SIZE = 1
EMBEDDING_LENGTH = 20
vocab = ["0", "1"]
# vocab = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d"]
symbol_to_ix = {symbol: i for i, symbol in enumerate(vocab)}
LIMIT_REGEX = 0


def upload_data(challenge=""):
    with open("train" + challenge, mode="r", encoding="utf-8") as f:
        x_train = []
        y_train = []
        for line in f:
            sample = []
            for i in range(len(line) - 3):
                sample.append(symbol_to_ix[line[i]])
            if len(sample) < LIMIT_REGEX:
                sample += [symbol_to_ix["<eos>"]] * (LIMIT_REGEX - len(sample))
            x_train.append(sample)
            y_train.append([int(line[-2])])
    with open("test" + challenge, mode="r", encoding="utf-8") as f:
        x_test = []
        y_test = []
        for line in f:
            sample = []
            for i in range(len(line) - 3):
                sample.append(symbol_to_ix[line[i]])
            if len(sample) < LIMIT_REGEX:
                sample += [symbol_to_ix["<eos>"]] * (LIMIT_REGEX - len(sample))
            x_test.append(sample)
            y_test.append([int(line[-2])])

    return x_train, y_train, x_test, y_test


class lstm_cell(nn.Module):
    def __init__(self, vocab_size, hidden_size, hidden_size_mlp, device):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, EMBEDDING_LENGTH)
        self.cell = nn.LSTM(EMBEDDING_LENGTH, hidden_size, 2)
        self.linear = nn.Linear(hidden_size, hidden_size_mlp)
        self.linear2 = nn.Linear(hidden_size_mlp, 1)
        self.device_to = device

    def forward(self, input):
        embd = self.embeddings(input.T)
        output, _ = self.cell(embd)
        lin1 = torch.tanh(self.linear(output[-1, :, :]))
        output = torch.sigmoid(self.linear2(lin1))
        return output


def train(x_train, y_train, x_test, y_test, challenge, device=torch.device("cpu"), num_iterations=1000, hidden_size=20, lr=0.1):
    our_model = lstm_cell(len(vocab), hidden_size, hidden_size, device)
    our_model: lstm_cell
    our_model.to(device)
    loss_func = nn.BCELoss()
    optimizer = torch.optim.SGD(our_model.parameters(), lr=lr)
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for epoch in range(num_iterations):
        total_loss = 0
        total_acc = 0
        for batch in range(0, len(x_train)):
            x_batch = torch.LongTensor(x_train[batch:batch + 1]).to(device)
            y_batch = torch.LongTensor(y_train[batch:batch + 1]).float().to(device)
            our_model.zero_grad()
            y_hat_prob = our_model.forward(x_batch).to(device)
            y_hat = 1.0 * (y_hat_prob > 0.5)
            total_acc += (y_hat == y_batch).int().sum()
            loss = loss_func(y_hat_prob, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_acc.append(total_acc.item() * (100 / (len(x_train))))
        train_loss.append(total_loss)
        total_acc = 0
        total_loss = 0
        with torch.no_grad():
            for batch in range(0, len(x_test)):
                x_batch = torch.LongTensor(x_test[batch:batch + 1]).to(device)
                y_batch = torch.LongTensor(y_test[batch:batch + 1]).float().to(device)
                our_model.zero_grad()
                y_hat_prob = our_model.forward(x_batch).to(device)
                y_hat = 1.0 * (y_hat_prob > 0.5)
                total_acc += (y_hat == y_batch).int().sum()
                loss = loss_func(y_hat_prob, y_batch)
                total_loss += loss.item()
            test_acc.append(total_acc.item() * (100 / (len(x_test))))
            test_loss.append(total_loss)
        print("epoch: {} train loss: {} train acc: {} test loss: {} test acc: {}".format(epoch, train_loss[-1], train_acc[-1], test_loss[-1], test_acc[-1]))
    plt.figure(1)
    plt.plot(train_acc)
    plt.title("{} - Train Accuracy".format(challenge))
    plt.savefig("{} Train Accuracy.jpg".format(challenge))
    plt.close()
    plt.figure(2)
    plt.plot(train_loss)
    plt.title("{} - Train Loss".format(challenge))
    plt.savefig("{} Train Loss.jpg".format(challenge))
    plt.close()
    plt.figure(3)
    plt.plot(test_acc)
    plt.title("{} - Test Accuracy".format(challenge))
    plt.savefig("{} Test Accuracy.jpg".format(challenge))
    plt.close()
    plt.figure(4)
    plt.plot(test_loss)
    plt.title("{} - Test Loss".format(challenge))
    plt.savefig("{} Test Loss.jpg".format(challenge))
    plt.close()

if __name__=="__main__":
    challenges = ["_start_end_same", "_Palindroms", "_primes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for challenge in challenges:
        x_train, y_train, x_test, y_test = upload_data(challenge)
        train(x_train, y_train, x_test, y_test, challenge, device=device, num_iterations=50, hidden_size=50, lr=1e-1)
