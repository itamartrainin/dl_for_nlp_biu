import mlp1 as ll
import random
import numpy as np

STUDENT={'name': 'Itamar Trainin',
         'ID': '315425967'}

def feats_to_vec(features):
    #Generate one-hot-vector for each bi-gram then sum all the vectors to one counted-hot-vector.
    one_hot_vec = np.zeros(utils.in_dim)
    for f in features:
        one_hot_vec[utils.F2I[f]] += 1
    # return one_hot_vec
    # return one_hot_vec / np.max(one_hot_vec)
    return (one_hot_vec * 30) / len(features) #best so far
    # return one_hot_vec * 30 / np.linalg.norm(one_hot_vec)

def make_prediction(features, params):
    x = feats_to_vec(features)

    joined_text = ''.join(features)
    pred = -1
    if 'œ' in joined_text or 'æ' in joined_text or 'ç' in joined_text:
        pred = utils.L2I['fr']
    elif 'ñ' in joined_text or '¿' in joined_text or '¡' in joined_text:
        pred = utils.L2I['es']
    elif 'ß' in joined_text:
        pred = utils.L2I['de']
    else:
        pred = ll.predict(x, params)

    return pred

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        y = utils.L2I[label]
        pred = make_prediction(features,params)

        if pred == y:
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

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)
            y = utils.L2I[label]
            loss, grads = ll.loss_and_gradients(x, y, params)
            cum_loss += loss
            for i in range(len(params)):
                params[i] = params[i] - learning_rate * grads[i]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params

def train_bigrams():
    in_dim = utils.in_dim
    out_dim = utils.out_dim
    hidden_dim = 100

    num_iterations = 1000
    learning_rate = 0.05

    # num_iterations = 100
    # learning_rate = 0.01

    train_data = utils.TRAIN
    dev_data = utils.DEV

    params = ll.create_classifier(in_dim, hidden_dim, out_dim)

    params[0] = np.abs(np.random.randn(params[0].shape[0], params[0].shape[1]))
    params[1] = np.abs(np.random.randn(params[1].shape[0]))
    params[2] = np.abs(np.random.randn(params[2].shape[0], params[2].shape[1]))
    params[3] = np.abs(np.random.randn(params[3].shape[0]))

    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

    test_evaluation(trained_params)

if __name__ == '__main__':
    import utils
    train_bigrams()
