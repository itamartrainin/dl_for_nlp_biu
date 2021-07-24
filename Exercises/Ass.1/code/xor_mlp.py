import mlp1 as mlp1
import random
import numpy as np

STUDENT={'name': 'Itamar Trainin',
         'ID': '315425967'}

def make_prediction(features, params):
    pred = mlp1.predict(features, params)
    return pred

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        pred = make_prediction(features,params)

        if pred == label:
            good += 1
        else:
            bad += 1
    return good / (good + bad)

def test_evaluation(params):
    TEST = [(l,utils.text_to_bigrams(t)) for l,t in utils.read_data("test")]
    fpredictions = open('test.pred_saved', 'w', encoding='utf-8')

    for label, features in TEST:
        pred = make_prediction(features, params)
        fpredictions.write(utils.LI2L[pred] + '\n')

def train_classifier(train_data, num_iterations, learning_rate, params):
    for I in range(num_iterations):
        cum_loss = 0.0   # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = features
            y = label
            loss, grads = mlp1.loss_and_gradients(x, y, params)
            cum_loss += loss
            for i in range(len(params)):
                params[i] = params[i] - learning_rate * grads[i]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        print(I, train_loss, train_accuracy)
    return params

def train_xor():
    in_dim = 2
    hidden_dim = 4
    out_dim = 2

    num_iterations = 100
    learning_rate = 0.1

    import xor_data
    train_data = xor_data.data

    params = mlp1.create_classifier(in_dim, hidden_dim, out_dim)

    params[0] = np.abs(np.random.randn(params[0].shape[0], params[0].shape[1]))
    params[1] = np.abs(np.random.randn(params[1].shape[0]))
    params[2] = np.abs(np.random.randn(params[2].shape[0], params[2].shape[1]))
    params[3] = np.abs(np.random.randn(params[3].shape[0]))

    trained_params = train_classifier(train_data, num_iterations, learning_rate, params)

if __name__ == '__main__':
    import utils
    train_xor()