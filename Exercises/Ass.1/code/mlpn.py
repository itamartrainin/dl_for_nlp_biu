import numpy as np

STUDENT={'name': 'Itamar Trainin',
         'ID': '315425967'}

def lin(x, params):
    W, b = params
    return np.dot(x, W) + b

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1 - tanh(x) ** 2

def classifier_output(x, params):
    x = np.array(x)
    num_params = len(params)

    if num_params < 2 or num_params % 2 == 1:
        print('Invalid params.')
        return

    activation_layer = x
    i = 0
    while i < num_params - 2:
        lin_layer = lin(activation_layer, [params[i], params[i+1]])
        t_i = tanh(lin_layer/np.max(lin_layer))
        # t_i = tanh(lin_layer)
        activation_layer = t_i
        i += 2
    probs = softmax(lin(activation_layer, [params[num_params - 2], params[num_params - 1]]))
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...
    """
    x = np.array(x)
    num_params = len(params)
    g = np.copy(params)

    if num_params < 2 or num_params % 2 == 1:
        print('Invalid params.')
        return

    t = [x]
    activation_layer = x
    i = 0
    while i < num_params - 2:
        lin_layer = lin(activation_layer, [params[i], params[i+1]])
        t_i = tanh(lin_layer/np.max(lin_layer))
        # t_i = tanh(lin_layer)
        t.append(t_i)
        activation_layer = t_i
        i += 2
    probs = softmax(lin(activation_layer, [params[num_params - 2], params[num_params - 1]]))
    t.append(0)

    loss = -np.log(probs[y])

    #top layer gradients
    dls_dl = np.copy(probs)
    dls_dl[y] -= 1

    g[num_params - 1] = dls_dl
    g[num_params - 2] = np.array([g[num_params - 1]]).T.dot(np.array([activation_layer])).T
    i = (num_params-2) - 1
    while i > 0:
        g[i] = np.dot(params[i+1], g[i+2]) * (1 - t[int((i-1)/2) + 1] ** 2)
        g[i-1] = np.array([g[i]]).T.dot(np.array([t[int((i-1)/2)]])).T
        i -= 2
    return loss, g

def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    for i in range(len(dims) - 1):
        W = np.abs(np.random.randn(dims[i], dims[i+1]))
        params.append(W)
        b = np.abs(np.random.randn(dims[i+1]))
        params.append(b)
    return params

if __name__ == '__main__':
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W0, b0, W1, b1, W2, b2 = create_classifier([20, 30, 40, 10])
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def _loss_and_W0_grad(W0):
        global b0
        global W1
        global b1
        global W2
        global b2
        loss, grads = loss_and_gradients(x, 0, [W0, b0, W1, b1, W2, b2])
        return loss, grads[0]

    def _loss_and_b0_grad(b0):
        global W0
        global W1
        global b1
        global W2
        global b2
        loss, grads = loss_and_gradients(x, 0, [W0, b0, W1, b1, W2, b2])
        return loss, grads[1]

    def _loss_and_W1_grad(W1):
        global W0
        global b0
        global b1
        global W2
        global b2
        loss, grads = loss_and_gradients(x, 0, [W0, b0, W1, b1, W2, b2])
        return loss, grads[2]

    def _loss_and_b1_grad(b1):
        global W0
        global b0
        global W1
        global W2
        global b2
        loss, grads = loss_and_gradients(x, 0, [W0, b0, W1, b1, W2, b2])
        return loss, grads[3]

    def _loss_and_W2_grad(W2):
        global W0
        global b0
        global W1
        global b1
        global b2
        loss, grads = loss_and_gradients(x, 0, [W0, b0, W1, b1, W2, b2])
        return loss, grads[4]

    def _loss_and_b2_grad(b2):
        global W0
        global b0
        global W1
        global b1
        global W2
        loss, grads = loss_and_gradients(x, 0, [W0, b0, W1, b1, W2, b2])
        return loss, grads[5]

    for _ in range(10):
        W0 = np.random.randn(W0.shape[0], W0.shape[1])
        b0 = np.random.randn(b0.shape[0])
        W1 = np.random.randn(W1.shape[0], W1.shape[1])
        b1 = np.random.randn(b1.shape[0])
        W2 = np.random.randn(W2.shape[0], W2.shape[1])
        b2 = np.random.randn(b2.shape[0])
        gradient_check(_loss_and_W0_grad, W0)
        gradient_check(_loss_and_b0_grad, b0)
        gradient_check(_loss_and_W1_grad, W1)
        gradient_check(_loss_and_b1_grad, b1)
        gradient_check(_loss_and_W2_grad, W2)
        gradient_check(_loss_and_b2_grad, b2)

    W, b, U, b_tag = create_classifier([3, 6, 4])

    def _loss_and_W_grad(W):
        global b
        global U
        global b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[0]

    def _loss_and_b_grad(b):
        global W
        global U
        global b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[1]

    def _loss_and_U_grad(U):
        global W
        global b
        global b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[2]

    def _loss_and_b_tag_grad(b_tag):
        global W
        global U
        global b
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[3]

    for _ in range(10):
        W = np.random.randn(W.shape[0], W.shape[1])
        b = np.random.randn(b.shape[0])
        U = np.random.randn(U.shape[0], U.shape[1])
        b_tag = np.random.randn(b_tag.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_U_grad, U)
        gradient_check(_loss_and_b_tag_grad, b_tag)