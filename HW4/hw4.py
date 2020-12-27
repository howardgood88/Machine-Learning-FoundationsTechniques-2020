# Answer: baedc
import numpy as np
from liblinearutil import *
import math

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
            train_y.append(1 if data[-1] > 0 else 0)
    return np.array(train_x), np.array(train_y)


def insert_x_0(x, x_0):
    '''
        Insert x_0 into the zero-th dimension of x_n.
    '''
    return np.hstack((
                np.array([ [x_0] for _ in range(len(x)) ]), 
                x
            ))


def transformQ(x, Q):
    '''
        Second-order polynomial transformation.
    '''
    new_x = []
    for sample in x:
        t = []
        for i in range(len(sample)):
            for j in range(i, len(sample)):
                t.append(sample[i] * sample[j])
        new_x.append(t)
    new_x = np.array(new_x)
    return np.hstack((
                x,
                new_x
            ))


def choose_val_for_cross_validation(val_num, train_x, train_y):
    '''
        Split train data by cross validation.
    '''
    return np.vstack((
                train_x[:val_num * 40],
                train_x[(val_num + 1) * 40:]
            )),\
            np.hstack((
                train_y[:val_num * 40],
                train_y[(val_num + 1) * 40:]
            )),\
            train_x[val_num * 40:(val_num + 1) * 40],\
            train_y[val_num * 40:(val_num + 1) * 40]

# ------------------------------------------------------------------------
#                             Loss function
# ------------------------------------------------------------------------

def zero_one(y, y_hat):
    diff = abs(y_hat - y)
    return diff.mean()

# ------------------------------------------------------------------------
#                                Problems
# ------------------------------------------------------------------------

def Problem_16(train_x, train_y, test_x, test_y):

    minEout = float('inf')
    # Train
    for lambda_log in (-4, -2, 0, 2, 4):
        C = 1 / 10 ** lambda_log / 2
        model = train(train_y, train_x, '-s 0 -c ' + str(C) + ' -e 0.000001 -q')

        p_label, p_acc, p_val = predict(test_y, test_x, model, '-b 1 -q')
        Eout = zero_one(test_y, p_label)
        
        if Eout == minEout:
            best_lambda_log = max(best_lambda_log, lambda_log)
        elif Eout < minEout:
            minEout = Eout
            best_lambda_log = lambda_log

    print('[Problem 16] log_10(lambda*):', best_lambda_log)


def Problem_17(train_x, train_y, test_x, test_y):

    minEin = float('inf')
    # Train
    for lambda_log in (-4, -2, 0, 2, 4):
        C = 1 / 10 ** lambda_log / 2
        model = train(train_y, train_x, '-s 0 -c ' + str(C) + ' -e 0.000001 -q')

        p_label, p_acc, p_val = predict(train_y, train_x, model, '-b 1 -q')
        Ein = zero_one(train_y, p_label)

        if Ein == minEin:
            best_lambda_log = max(best_lambda_log, lambda_log)
        elif Ein < minEin:
            minEin = Ein
            best_lambda_log = lambda_log

    print('[Problem 17] log_10(lambda*):', best_lambda_log)


def Problem_18_19(train_x, train_y, test_x, test_y):
    # Split train_x to train and val
    train_x_split, train_y_split = train_x[:120], train_y[:120]
    val_x, val_y = train_x[120:], train_y[120:]

    minEval = float('inf')
    # Train
    for lambda_log in (-4, -2, 0, 2, 4):
        C = 1 / 10 ** lambda_log / 2
        model = train(train_y_split, train_x_split, '-s 0 -c ' + str(C) + ' -e 0.000001 -q')

        p_label, p_acc, p_val = predict(val_y, val_x, model, '-b 1 -q')
        Eval = zero_one(val_y, p_label)

        if Eval == minEval and lambda_log > best_lambda_log:
            best_lambda_log = lambda_log
            best_model = model
        elif Eval < minEval:
            minEval = Eval
            best_lambda_log = lambda_log
            best_model = model

    # print('best_lambda_log:', best_lambda_log)
    p_label, p_acc, p_val = predict(test_y, test_x, best_model, '-b 1 -q')
    Eout = zero_one(test_y, p_label)

    print('[Problem 18] Eout(w^-_lambda*):', Eout)

    C = 1 / 10 ** best_lambda_log / 2
    model = train(train_y, train_x, '-s 0 -c ' + str(C) + ' -e 0.000001 -q')

    p_label, p_acc, p_val = predict(test_y, test_x, model, '-b 1 -q')
    Eout = zero_one(test_y, p_label)

    print('[Problem 19] Eout(w_lambda*):', Eout)


def Problem_20(train_x, train_y, test_x, test_y):
    V = 5

    minEcv = float('inf')
    # Train
    for lambda_log in (-4, -2, 0, 2, 4):
        Ecv = 0
        C = 1 / 10 ** lambda_log / 2
        for i in range(V):
            train_x_split, train_y_split, val_x, val_y = choose_val_for_cross_validation(i, train_x, train_y)
            model = train(train_y_split, train_x_split, '-s 0 -c ' + str(C) + ' -e 0.000001 -q')

            p_label, p_acc, p_val = predict(val_y, val_x, model, '-b 1 -q')
            Eval = zero_one(val_y, p_label)
            Ecv += Eval

        Ecv = Ecv / V
        if Ecv == minEcv and lambda_log > best_lambda_log:
            best_lambda_log = lambda_log
            best_Ecv = Ecv
        elif Ecv < minEcv:
            minEcv = Ecv
            best_lambda_log = lambda_log
            best_Ecv = Ecv

    print('[Problem 20] Ecv(A_lambda*):', best_Ecv)

# ------------------------------------------------------------------------
#                                   Main
# ------------------------------------------------------------------------

if __name__ == '__main__':
    # Parameters setting
    para = {
        'x_0' : 1,
        'Q' : 2,
    }

    # Read data
    train_dir = 'train.dat' # Please change to your own train.dat dir
    test_dir = 'test.dat'
    train_x, train_y = read_data(train_dir)
    test_x, test_y = read_data(test_dir)
    print('Shape of raw data x:', train_x.shape)

    # Transform
    train_x = transformQ(train_x, para['Q'])
    test_x = transformQ(test_x, para['Q'])
    
    # Insert x_0 into train_x
    train_x = insert_x_0(train_x, para['x_0'])
    test_x = insert_x_0(test_x, para['x_0'])

    print('Train x size:', train_x.shape, 'train y size:', train_y.shape)
    print('Test x size:', test_x.shape, 'Test y size:', test_y.shape)
    print()

    # Problem 16 to 20
    Problem_16(train_x, train_y, test_x, test_y)
    Problem_17(train_x, train_y, test_x, test_y)
    Problem_18_19(train_x, train_y, test_x, test_y)
    Problem_20(train_x, train_y, test_x, test_y)