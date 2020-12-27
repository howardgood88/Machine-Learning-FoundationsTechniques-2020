# Answer: baedc
import numpy as np
from svmutil import *
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from sklearn.utils import shuffle

# ------------------------------------------------------------------------
#                               Utilities
# ------------------------------------------------------------------------

def label_data(input_, class_):
    output = []
    for i in input_:
        output.append(1 if i == class_ else -1)
    return np.array(output)


def get_random_val(x, y, N = 200):
    x, y = shuffle(x, y)
    return x[200:], y[200:], x[:200], y[:200]

# ------------------------------------------------------------------------
#                                Problems
# ------------------------------------------------------------------------

def Problem_15(train_x, train_y):
    para = {
        'C' : 10,
        'class' : 3,
        }

    train_y_ = label_data(train_y, class_ = para['class'])

    model = SVC(kernel = 'linear', C = para['C'])
    model.fit(train_x, train_y_)
    print('[Problem 15] ||w||:', np.linalg.norm(model.coef_.toarray()))


def Problem_16_17(train_x, train_y):
    para = {
        'C' : 10,
        'Q' : 2,
        }

    best_class_accu = 0
    max_nr_sv = 0
    for class_ in range(1, 6):
        train_y_ = label_data(train_y, class_ = class_)

        model = svm_train(train_y_, train_x, '-t 1 -d ' + str(para['Q']) + ' -g 1 -r 1 -c ' + str(para['C']) + ' -q')
        p_label, p_acc, p_val = svm_predict(train_y_, train_x, model, '-q')
        if p_acc[0] > best_class_accu:
            best_class_accu = p_acc[0]
            best_class = class_
        max_nr_sv = max(max_nr_sv, model.get_nr_sv())
    
    print('[Problem 16] class:', best_class)
    print('[Problem 17] nr sv:', max_nr_sv)


def Problem_18(train_x, train_y, test_x, test_y, class_ = 6):
    para = {
        'class' : 6,
        }

    train_y_ = label_data(train_y, class_ = para['class'])
    test_y_ = label_data(test_y, class_ = para['class'])

    max_accu = 0
    best_C = []
    for C in (0.01, 0.1, 1, 10, 100):
        model = svm_train(train_y_, train_x, '-t 2 -g 10 -c ' + str(C) + ' -q')
        p_label, p_acc, p_val = svm_predict(test_y_, test_x, model, '-q')

        if p_acc[0] > max_accu:
            max_accu = p_acc[0]
            best_C = [C]
        elif p_acc[0] == max_accu:
            best_C.append(C)
    
    print('[Problem 18] Best C:', best_C)


def Problem_19(train_x, train_y, test_x, test_y, class_ = 6):
    para = {
        'C' : 0.1,
        'class' : 6,
        }

    train_y_ = label_data(train_y, class_ = para['class'])
    test_y_ = label_data(test_y, class_ = para['class'])

    max_accu = 0
    best_r = []
    for r in (0.1, 1, 10, 100, 1000):
        model = svm_train(train_y_, train_x, '-t 2 -g ' + str(r) + ' -c ' + str(para['C']) + ' -q')
        p_label, p_acc, p_val = svm_predict(test_y_, test_x, model, '-q')

        if p_acc[0] > max_accu:
            max_accu = p_acc[0]
            best_r = [r]
        elif p_acc[0] == max_accu:
            best_r.append(r)
    
    print('[Problem 19] Best r:', best_r)


def Problem_20(train_x, train_y):
    para = {
        'C' : 0.1,
        'class' : 6,
        }

    train_y_ = label_data(train_y, class_ = para['class'])

    times_r = {0.1:0, 1:0, 10:0, 100:0, 1000:0}
    for i in range(1000):
        train_x_, train_y_, val_x, val_y = get_random_val(train_x, train_y)
        max_accu = 0
        for r in (0.1, 1, 10, 100, 1000):
            model = svm_train(train_y_, train_x_, '-t 2 -g ' + str(r) + ' -c ' + str(para['C']) + ' -q')
            p_label, p_acc, p_val = svm_predict(val_y, val_x, model, '-q')

            if p_acc[0] > max_accu:
                max_accu = p_acc[0]
                best_r = r
        times_r[best_r] += 1
        if i % 100 == 0:
            print(i)
    print(times_r)
    max_times = 0
    for r, times in times_r.items():
        if times > max_times:
            max_times = times
            best_r = r

    print('[Problem 20] Best r most:', best_r)

# ------------------------------------------------------------------------
#                                   Main
# ------------------------------------------------------------------------

if __name__ == '__main__':
    # Read data
    train_dir = 'train.dat' # Please change to your own train.dat dir
    test_dir = 'test.dat'
    train_x, train_y = load_svmlight_file(train_dir)
    test_x, test_y = load_svmlight_file(test_dir)

    print('Train x size:', train_x.shape, 'train y size:', train_y.shape)
    print('Test x size:', test_x.shape, 'Test y size:', test_y.shape)
    print()

    # Problem 16 to 20
    Problem_15(train_x, train_y)
    Problem_16_17(train_x, train_y)
    Problem_18(train_x, train_y, test_x, test_y)
    Problem_19(train_x, train_y, test_x, test_y)
    Problem_20(train_x, train_y)