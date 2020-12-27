# dccbabd
import random
import numpy as np
from statistics import median

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


def print_problem_number(problem_number):
    '''
        Print the problem number from "problem_number".
    '''
    def decorate(func):
        def decorate2(*args, **argv):
            print('[Problem {}]'.format(decorate.calls), end = ' ')
            decorate.calls += 1
            func(*args, **argv)
        decorate.calls = problem_number
        return decorate2
    return decorate


def insert_x_0(x, x_0, N):
    '''
        Insert x_0 into the zero-th dimension of x_n.
    '''
    return np.hstack((
                np.array([ [x_0] for _ in range(N) ]), 
                x
            ))


def get_random_number(N):
    '''
        Get random "int" from [0, N).
    '''
    return random.randint(0, N - 1)


def random_select_one(x, y):
    '''
        Random select one sample from array x and y.
    '''
    num = get_random_number(len(x))
    return x[num], y[num]


def sign(x):
    '''
        Take sign() to all element in array x. ( <0 as 1, >0 as 0)
    '''
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = 1
        else:
            x[i] = 0
    return x


def get_w_Lin(train_x, train_y):
    ''' 
        pseudo inverse to get w_Lin
    '''
    pseudo_inverse_x = np.linalg.pinv(train_x)
    return np.dot(pseudo_inverse_x, train_y)


def transformQ(x, Q):
    '''
        Homogeneous order-Q polynomial transform.
    '''
    new_x = []
    for k in range(2, Q + 1):
        new_x.append([[pow(j, k) for j in i] for i in x])
    new_x = np.array(new_x)
    return np.hstack((
                x,
                *new_x
            ))


def logistic(x):
    '''
        Logistic function.
    '''
    return 1 / (1 + np.exp(-x))

# ------------------------------------------------------------------------
#                             Loss function
# ------------------------------------------------------------------------

def MSE(x, y, w):
    return np.square(np.dot(x, w) - y).mean()


def CE(x, y, w):
    return (-np.log(logistic(np.dot(x, w) * y))).mean()


def zero_one(x, y, w):
    return sign(np.dot(x, w) * y).mean()

# ------------------------------------------------------------------------
#                             Training Model
# ------------------------------------------------------------------------

def SGD_linear(train_x, train_y, para):
    # initalize w
    w = np.zeros(len(train_x[0]))

    Ein = float('inf')
    iter_times = 0
    while Ein > para['stop_iter']:
        x, y = random_select_one(train_x, train_y)
        w = w + para['eta'] * 2 * (y - np.dot(x, w)) * x
        Ein = MSE(train_x, train_y, w)
        iter_times += 1

    return iter_times


def SGD_logistic(train_x, train_y, w, para):
    for _ in range(para['stop_iter']):
        x, y = random_select_one(train_x, train_y)
        w = w + para['eta'] * logistic(-np.dot(x, w) * y) * y * x
        Ein = CE(train_x, train_y, w)
    return Ein


def linear_regression(train_x, train_y, loss_func = MSE):
    w_Lin = get_w_Lin(train_x, train_y)
    Ein = loss_func(train_x, train_y, w_Lin)
    return w_Lin, Ein

# ------------------------------------------------------------------------
#                                Problems
# ------------------------------------------------------------------------

def Problem_14(train_x, train_y, para):
    # Insert x_0 into train_x
    train_x = insert_x_0(train_x, para['x_0'], para['train_size'])

    # Train
    w_Lin, Ein = linear_regression(train_x, train_y, loss_func = MSE)

    print('[Problem 14] Ein:', Ein)
    return w_Lin, Ein


def Problem_15(train_x, train_y, Ein, para):
    # Parameters setting
    para['stop_iter'] = 1.01 * Ein
    
    # Insert x_0 into train_x
    train_x = insert_x_0(train_x, para['x_0'], para['train_size'])

    # Training
    iters = []
    for _ in range(para['Epoch']):
        iter_times = SGD_linear(train_x, train_y, para)
        iters.append(iter_times)
    avg_iter = sum(iters) / len(iters)

    print('[Problem 15] avg iteration times:', avg_iter)


@print_problem_number(problem_number = 16)
def Problem_16_17(train_x, train_y, para, w0 = 0):
    # Parameters setting
    para['stop_iter'] = 500
    
    # Insert x_0 into train_x
    train_x = insert_x_0(train_x, para['x_0'], para['train_size'])

    # Training
    Eins = []
    for _ in range(para['Epoch']):
        Ein = SGD_logistic(train_x, train_y, w0, para)
        Eins.append(Ein)
    avg_Eins = sum(Eins) / len(Eins)

    print('avg Ein:', avg_Eins)


def Problem_18(train_x, train_y, test_x, test_y, para):
    # Insert x_0 into train_x
    train_x = insert_x_0(train_x, para['x_0'], para['train_size'])
    test_x = insert_x_0(test_x, para['x_0'], para['test_size'])

    # Train
    w_Lin, Ein = linear_regression(train_x, train_y, loss_func = zero_one)
    Eout = zero_one(test_x, test_y, w_Lin)

    print('[Problem 18] |Ein - Eout|:', abs(Ein - Eout))


@print_problem_number(problem_number = 19)
def Problem_19_20(train_x, train_y, test_x, test_y, para, Q = 3):
    # Transform
    train_x = transformQ(train_x, Q)
    test_x = transformQ(test_x, Q)

    # Insert x_0 into train_x
    train_x = insert_x_0(train_x, para['x_0'], para['train_size'])
    test_x = insert_x_0(test_x, para['x_0'], para['test_size'])

    # Train
    w_Lin, Ein = linear_regression(train_x, train_y, loss_func = zero_one)
    Eout = zero_one(test_x, test_y, w_Lin)

    print('|Ein - Eout|:', abs(Ein - Eout))

# ------------------------------------------------------------------------
#                                   Main
# ------------------------------------------------------------------------

if __name__ == '__main__':
    # Read data
    train_dir = 'train.dat' # Please change to your own train.dat dir
    test_dir = 'test.dat'
    train_x, train_y = read_data(train_dir)
    test_x, test_y = read_data(test_dir)

    # Parameters setting
    Parameters = {
        'train_size' : len(train_x),
        'test_size' : len(test_x),
        'Epoch' : 1000,
        'x_0' : 1,
        'eta' : 0.001,
    }

    # Problem 16 to 20
    w_Lin, Ein = Problem_14(train_x, train_y, Parameters)
    Problem_15(train_x, train_y, Ein, Parameters)
    Problem_16_17(train_x, train_y, Parameters, w0 = 0)
    Problem_16_17(train_x, train_y, Parameters, w0 = w_Lin)
    Problem_18(train_x, train_y, test_x, test_y, Parameters)
    Problem_19_20(train_x, train_y, test_x, test_y, Parameters, Q = 3)
    Problem_19_20(train_x, train_y, test_x, test_y, Parameters, Q = 10)