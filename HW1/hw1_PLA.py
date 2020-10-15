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

        Parameters:
        filename (str): Filename of full directory

        Returns:
        x_data (np array): List of input in numpy array
        y_data (np array): List of label in numpy array
    '''

    x_data, y_data = [], []
    with open(filename) as f:
        for line in f.readlines():
            data = [float(i) for i in line.split()]
            x_data.append(data[:-1])
            y_data.append(data[-1])
    return np.array(x_data), np.array(y_data)


def insert_x_0(x_data, x_0, N):
    '''
        Insert x_0 into the zero-th dimension of x_n.

        Parameters:
        x_data (np array): List of input in numpy array
        x_0 (int): x_0 (-threshold)
        N (int): number of input

        Returns:
        - (np array): List of input after insert x_0 in numpy array
    '''

    return np.hstack((
                np.array([ [x_0] for _ in range(N) ]), 
                x_data
            ))


def get_random_number(N):
    '''
        Get a random number int range [0, N).

        Parameters:
        N (int): range of number

        Returns:
        - (int): random number
    '''

    return random.randint(0, N - 1)


def dot_(w, x):
    '''
        Dot of vector w and x, w^Tx in math form.

        Parameters:
        w (np array): vector 1
        x (np array): vector 2

        Returns:
        - (int): dot
    '''

    return np.dot(w.transpose(), x)

# ------------------------------------------------------------------------
#                                Training
# ------------------------------------------------------------------------

def PLA(x_data, y_data, N, Stop_iter):
    '''
        PLA algorithm.

        Parameters:
        x_data (np array): List of input in numpy array
        y_data (list): List of label in numpy array
        Stop_iter (int): Stop training when correct consecutively over
                         Stop_iter iterations

        Returns:
        update_count (int): Times of update
        w[0] (int): w_0 (-Threshold)
    '''

    # Initialize w
    w = np.zeros(len(x_data[0]))

    # Start training
    correct_count = 0
    update_count = 0
    while(correct_count < Stop_iter):
        randPick = get_random_number(N)
        randPick_x, randPick_y = x_data[randPick], y_data[randPick]
        dot = dot_(w, randPick_x) * randPick_y
        if (dot <= 0):
            w = w + randPick_y * randPick_x
            correct_count = 0
            update_count += 1
        else:
            correct_count += 1
    return update_count, w[0]


def Train(x_data, y_data, para):
    '''
        Main function of training.

        Parameters:
        x_data (np array): List of input in numpy array
        y_data (np array): List of label in numpy array
        para (dict): Dictionary of parameter

        Returns:
        updates (list): List of update times of all epoch
        w0s (list): List of w0 of all epoch
    '''

    updates, w0s = [], []
    for _ in range(para['Epoch']):
        update_count, w0 = PLA(x_data, y_data, para['N'], para['Stop_iter'])
        updates.append(update_count)
        w0s.append(w0)
    return updates, w0s

# ------------------------------------------------------------------------
#                                Problems
# ------------------------------------------------------------------------

def Problem_16_17(x_data, y_data, para):
    '''
        For problem 16 and 17, x_0 set to be 1. Find the Median of update times
        and Median of W_0s.

        Parameters:
        x_data (np array): List of input in numpy array
        y_data (np array): List of label in numpy array
        para (dict): Dictionary of parameter

        Returns:
    '''

    # Parameters setting
    para['x_0'] = 1

    # Initialization
    ## Insert x_0 into x_data
    x_data = insert_x_0(x_data, para['x_0'], para['N'])

    # Train
    updates, w0s = Train(x_data, y_data, para)

    print('[Problem 16] Median of update times:', median(updates))
    print('[Problem 17] Median of W_0s:', median(w0s))


def Problem_18(x_data, y_data, para):
    '''
        For problem 18, x_0 set to be 10. Find the Median of update times.

        Parameters:
        x_data (np array): List of input in numpy array
        y_data (np array): List of label in numpy array
        para (dict): Dictionary of parameter

        Returns:
    '''

    # Parameters setting
    para['x_0'] = 10

    # Initialization
    ## Insert x_0 into x_data
    x_data = insert_x_0(x_data, para['x_0'], para['N'])

    # Train
    updates, _ = Train(x_data, y_data, para)

    print('[Problem 18] Median of update times:', median(updates))


def Problem_19(x_data, y_data, para):
    '''
        For problem 19, x_0 set to be 0. Find the Median of update times.

        Parameters:
        x_data (np array): List of input in numpy array
        y_data (np array): List of label in numpy array
        para (dict): Dictionary of parameter

        Returns:
    '''

    # Parameters setting
    para['x_0'] = 0

    # Initialization
    ## Insert x_0 into x_data
    x_data = insert_x_0(x_data, para['x_0'], para['N'])

    # Train
    updates, _ = Train(x_data, y_data, para)

    print('[Problem 19] Median of update times:', median(updates))


def Problem_20(x_data, y_data, para):
    '''
        For problem 20, x_0 set to be 0, x_n scale down by 4. Find the 
        Median of update times.

        Parameters:
        x_data (np array): List of input in numpy array
        y_data (np array): List of label in numpy array
        para (dict): Dictionary of parameter

        Returns:
    '''

    # Parameters setting
    para['x_0'] = 0

    # Initialization
    ## Insert x_0 into x_data
    x_data = insert_x_0(x_data, para['x_0'], para['N'])
    ## Scale down by 4
    x_data = x_data / 4

    # Train
    updates, _ = Train(x_data, y_data, para)

    print('[Problem 20] Median of update times:', median(updates))

# ------------------------------------------------------------------------
#                                   Main
# ------------------------------------------------------------------------

if __name__ == '__main__':
    # Read data
    dir = 'train.dat' # Please change to your own train.dat dir
    x_data, y_data = read_data(dir)

    # Parameters setting
    Parameters = {
        'N' : len(x_data),
        'Epoch' : 1000,
        'Stop_iter' : 5 * 100,
    }

    # Problem 16 to 20
    Problem_16_17(x_data, y_data,Parameters)
    Problem_18(x_data, y_data, Parameters)
    Problem_19(x_data, y_data, Parameters)
    Problem_20(x_data, y_data, Parameters)