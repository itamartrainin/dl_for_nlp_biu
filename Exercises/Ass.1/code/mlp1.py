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

def classifier_output(x, params):
    W, b, U, b_tag = params
    first_layer = lin(x, [W, b])                        # Wx + b
    activation = tanh(first_layer/np.max(first_layer))  # tanh(Wx + b)
    # activation = tanh(first_layer)  # tanh(Wx + b)
    hidden_layer = lin(activation, [U, b_tag])          # U(tanh(Wx + b)) + b'
    probs = softmax(hidden_layer)                       # softmax(U(tanh(Wx + b)) + b')
    return probs

def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    W, b, U, b_tag = params
    x = np.array(x)

    # y_hat = classifier_output(x, params)
    # loss = -np.log(y_hat[y])

    f4 = lin(x, [W, b])
    f3 = tanh(f4/np.max(f4))
    # f3 = tanh(f4)
    f2 = lin(f3, [U, b_tag])
    f1 = -np.log(softmax(f2))

    #sanity check
    # if f1[y] != loss:
    #     print('Loss was not calculated correctly')
    #     return

    gf1_gf2 = softmax(f2)
    gf1_gf2[y] -= 1

    loss = f1[y]
    gb_tag = gf1_gf2
    gU = np.array([gb_tag]).T.dot(np.array([f3])).T
    gb = np.dot(U, gb_tag) * (1 - f3 ** 2)
    gW = np.array([gb]).T.dot(np.array([x])).T

    return loss, [gW, gb, gU, gb_tag]

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    W       = np.zeros((in_dim, hid_dim))
    b       = np.zeros(hid_dim)
    U       = np.zeros((hid_dim, out_dim))
    b_tag   = np.zeros(out_dim)
    return [W, b, U, b_tag]


if __name__ == '__main__':
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W, b, U, b_tag = create_classifier(3, 6, 4)

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
