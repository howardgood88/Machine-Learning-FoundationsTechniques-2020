# cdadb
import numpy as np
import random

# ------------------------------------------------------------------------
#                               Utilities
# ------------------------------------------------------------------------

def read_data(filename):
    '''
        Read data from the file, and the last element as label,
        the others as input in each line.
    '''
    train_x, train_y = [], []
    with open(filename) as f:
        for line in f.readlines():
            data = [float(i) for i in line.split()]
            train_x.append(data[:-1])
            train_y.append(data[-1])
    return np.array(train_x), np.array(train_y)


def random_select_one(x, y):
    '''
        Random select one sample from array x and y.
    '''
    num = random.randint(0, len(x - 1))
    return x[num], y[num]


def sign(x):
    '''
        <0 as -1, >0 as 1.
    '''
    if x <= 0:
        return -1
    else:
        return 1


def make_theta_list(x):
    '''
        Make theta on each segment.
    '''
    theta_list = []
    for i in range(1, len(x)):
        if x[i] != x[i - 1]:
            theta_list.append((x[i] + x[i - 1]) / 2)
    return np.array(theta_list)


def decision_stump(x, y, theta):
    mask1 = x < theta
    mask2 = x > theta
    return np.array([y[mask1], y[mask2]])


def split_data_train(X, y, i, theta):
    '''
        Split all data by feature i and decision stump theta.
    '''
    mask1 = X[:, i] < theta
    mask2 = X[:, i] > theta
    return X[mask1], y[mask1], X[mask2], y[mask2]


def split_data_test(X, y, y_idx, i, theta):
    '''
        Split all data by feature i and decision stump theta (record index of y).
    '''
    mask1 = X[:, i] < theta
    mask2 = X[:, i] > theta
    return X[mask1], y[mask1], y_idx[mask1], X[mask2], y[mask2], y_idx[mask2]


def zero_one(y, y_hat):
    N = len(y)
    return (y != y_hat).sum() / N


# ------------------------------------------------------------------------
#                             Training Model
# ------------------------------------------------------------------------


def bagging(X, y, N):
    X_, y_ = [], []
    not_chosen = np.full(len(y), True)
    for _ in range(N):
        num = random.randint(0, len(y) - 1)
        X_.append(X[num])
        y_.append(y[num])
        not_chosen[num] = False
    return np.array(X_), np.array(y_), not_chosen


class Tree:
    def __init__(self, i, theta):
        self.left = None
        self.right = None
        self.i = i
        self.theta = theta
        self.label = None


def Gini(y):
    '''
        Calculate Gini index.
    '''
    s = 0
    for k in [1, -1]:
        s += ((y == k).sum() / len(y)) ** 2
    return 1 - s


def branching(X, y):
    '''
        Branching by decision stump.
    '''
    n_features = X.shape[1]
    best_impurity = float('inf')
    for i in range(n_features):
        x_, y_ = zip(*sorted(zip(X[:, i], y)))
        x_, y_ = np.array(x_), np.array(y_)
        theta_list = make_theta_list(x_)
        for theta in theta_list:
            s = 0
            y_list = decision_stump(x_, y_, theta)
            for c in range(para['C']):
                impurity = Gini(y_list[c])
                s += len(y_list[c]) * impurity

            if s < best_impurity:
                best_impurity = s
                best_i = i
                best_theta = theta
    return best_i, best_theta


def cant_branch(X, y):
    '''
        Termination criteria.
    '''
    if np.all(y == y[0]):
        return True
    if np.all(X == X[0]):
        return True
    return False


def predict(X, y, root, y_hat, y_idx):
    '''
        Predict the y_hat by X and y.
    '''
    if root.i == None:
        for i in y_idx:
            y_hat[i] = root.label
        return
    X1, y1, y_idx1, X2, y2, y_idx2 = split_data_test(X, y, y_idx, root.i, root.theta)
    predict(X1, y1, root.left, y_hat, y_idx1)
    predict(X2, y2, root.right, y_hat, y_idx2)


def CART(X, y):
    if cant_branch(X, y):
        node = Tree(None, None)
        node.label = y[0]
        return node
    i, theta = branching(X, y)
    # print(i, theta)
    node = Tree(i, theta)
    X1, y1, X2, y2 = split_data_train(X, y, i, theta)
    node.left = CART(X1, y1)
    node.right = CART(X2, y2)
    node.label = node.left.label + node.right.label
    return node

# ------------------------------------------------------------------------
#                                Problems
# ------------------------------------------------------------------------

def Problem_14(train_x, train_y, test_x, test_y):
    root = CART(train_x, train_y)
    N = len(train_y)
    y_hat = np.zeros(N)
    predict(test_x, test_y, root, y_hat, np.arange(N))

    print('[Problem 14]', zero_one(test_y, y_hat))


def Problem_15_to_18(train_x, train_y, test_x, test_y):
    N = len(train_y)
    total_Eout = 0
    G_in = np.zeros(N)
    G_out = np.zeros(N)
    G_minus_list = np.full(N, np.inf)
    for _ in range(para['n_tree']):
        bagging_x, bagging_y, not_chosen = bagging(train_x, train_y, N // 2)
        root = CART(bagging_x, bagging_y)

        # Problem 15
        y_hat = np.zeros(N)
        predict(test_x, test_y, root, y_hat, np.arange(N))
        total_Eout += zero_one(test_y, y_hat)

        # Problem 16
        y_hat_train = np.zeros(N)
        predict(train_x, train_y, root, y_hat_train, np.arange(N))
        G_in += y_hat_train

        # Problem 17
        y_hat = np.zeros(N)
        predict(test_x, test_y, root, y_hat, np.arange(N))
        G_out += y_hat

        # Problem 18
        for idx, y in zip(np.arange(N)[not_chosen], y_hat_train[not_chosen]):
            if G_minus_list[idx] == np.inf:
                G_minus_list[idx] = y
            else:
                G_minus_list[idx] += y

    print('[Problem 15]', total_Eout / para['n_tree'])

    y_hat = []
    for i in G_in:
        y_hat.append(sign(i))
    print('[Problem 16]', zero_one(train_y, y_hat))

    y_hat = []
    for i in G_out:
        y_hat.append(sign(i))
    print('[Problem 17]', zero_one(test_y, y_hat))

    G_minus_list = [sign(G_minus) if G_minus != np.inf else -1 for G_minus in G_minus_list]
    print('[Problem 18]', zero_one(train_y, G_minus_list))

# ------------------------------------------------------------------------
#                                   Main
# ------------------------------------------------------------------------

# Parameters setting
para = {
    'C' : 2,
    'n_tree' : 2000,
}

if __name__ == '__main__':
    # Read data
    train_dir = 'train.dat' # Please change to your own train.dat dir
    test_dir = 'test.dat'
    train_x, train_y = read_data(train_dir)
    test_x, test_y = read_data(test_dir)

    # Problem 14 to 18
    Problem_14(train_x, train_y, test_x, test_y)
    Problem_15_to_18(train_x, train_y, test_x, test_y)