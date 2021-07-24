import loglinear as ll
import random
import numpy as np

STUDENT={'name': 'Itamar Trainin',
         'ID': '315425967'}

def feats_to_vec(features):
    #Generate one-hot-vector for each bi-gram then sum all the vectors to one counted-hot-vector.
    one_hot_vec = np.zeros(utils.in_dim_uni)
    for f in features:
        one_hot_vec[utils.F2I_uni[f]] += 1
    return (one_hot_vec * 30) / len(features) #best so far

def make_prediction(features, params):
    x = feats_to_vec(features)

    joined_text = ''.join(features)
    pred = -1
    if 'œ' in joined_text or 'æ' in joined_text or 'ç' in joined_text:
        pred = utils.L2I_uni['fr']
    elif 'ñ' in joined_text or '¿' in joined_text or '¡' in joined_text:
        pred = utils.L2I_uni['es']
    elif 'ß' in joined_text:
        pred = utils.L2I_uni['de']
    else:
        pred = ll.predict(x, params)

    return pred

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        y = utils.L2I_uni[label]
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

def train_classifier_unigrams(train_data, dev_data, num_iterations, learning_rate, params):
    for I in range(num_iterations):
        cum_loss = 0.0   # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)
            y = utils.L2I_uni[label]
            loss, grads = ll.loss_and_gradients(x, y, params)
            cum_loss += loss
            for i in range(len(params)):
                params[i] = params[i] - learning_rate * grads[i]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params

def train_unigrams():
    in_dim = utils.in_dim_uni
    out_dim = utils.out_dim_uni
    hidden_dim = 100

    num_iterations = 100
    learning_rate = 0.005

    train_data = utils.TRAIN_uni
    dev_data = utils.DEV_uni

    params = ll.create_classifier(in_dim, out_dim)

    params[0] = np.abs(np.random.randn(params[0].shape[0], params[0].shape[1]))
    params[1] = np.abs(np.random.randn(params[1].shape[0]))

    trained_params = train_classifier_unigrams(train_data, dev_data, num_iterations, learning_rate, params)

if __name__ == '__main__':
    import utils
    train_unigrams()