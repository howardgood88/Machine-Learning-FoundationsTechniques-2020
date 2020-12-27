import random
import numpy as np
from statistics import median
from operator import add

# ------------------------------------------------------------------------
#                               Utilities
# ------------------------------------------------------------------------

def generate_data(num):
    '''
        Generate x and y data of size num.

        Parameters:
        num (int): size

        Returns:
        x_data (list): Generate by uniform rand num in [-1, 1]
        y_data (list): Label by f(x) = sign(x)
    '''

    x_data = [random.uniform(-1, 1) for i in range(num)]
    y_data = [1 if i > 0 else -1 for i in x_data]
    return x_data, y_data


def flip(y, tau):
    '''
        Flip label y by probability tau "in place".

        Parameters:
        y (int): y data
        tau (int): probability to flip

        Returns:
        -
    '''

    if tau == 0:
        return
    for idx, i in enumerate(y):
        if random.randint(1, tau * 100) == 1:
            y[idx] = -i


def count_y_label(y):
    '''
        Count the number of 1 and -1 respectively.

        Parameters:
        y (int): y data

        Returns:
        total_1 (int): Number of 1 in total
        total_minus_1 (int): Number of -1 in total
    '''

    total_1, total_minus_1 = 0, 0
    for i in y:
        if i > 0:
            total_1 += 1
        else:
            total_minus_1 += 1
    return total_1, total_minus_1


def print_problem_number(func):
    '''
        Print the problem number from 16.

        Parameters:
        func (function): function to decorate

        Returns:
        decorate (function): decorator
    '''

    def decorate(*args):
        print('[Problem {}] '.format(decorate.calls), end='')
        decorate.calls += 1
        func(*args)
    decorate.calls = 16
    return decorate

# ------------------------------------------------------------------------
#                                 Testing
# ------------------------------------------------------------------------

def Get_Eout(x_test, y_test, g_theta, g_s):
    '''
        Calculate Eout.

        Parameters:
        x_test, y_test (list): x, y data
        g_theta (int): theta of hypothesis g
        g_s (int): s of hypothesis g

        Returns:
        Eout (int): Eout
    '''

    err = 0
    for x, y in zip(x_test, y_test):
        err += 0 if (x - g_theta) * g_s * y > 0 else 1
    Eout = err / len(x_test)
    
    return Eout

# ------------------------------------------------------------------------
#                                Training
# ------------------------------------------------------------------------

def decision_stump(x_train, y_train):
    '''
        Algorithm of decision stump.

        Parameters:
        x_train, y_train (list): train data

        Returns:
        Ein (int): min Ein
        g_theta (int): theta under min Ein
        g_s (int): s under min Ein
    '''

    # (1) Sort
    x_train, y_train = zip(*sorted(zip(x_train, y_train)))

    # (2)(3)
    ## Generate theta
    thetas = [-1]
    for idx in range(1, len(x_train)):
        thetas.append((x_train[idx] + x_train[idx - 1]) / 2)

    ## Calculate total number of 1 and -1
    total_1, total_minus_1 = count_y_label(y_train)

    ## get min Ein and correspouding theta and s
    Ein = float('inf')
    dp = [[0, 0] for i in range(len(y_train))]  # dp[i][0] for 1's of the left, dp[i][1] for -1's
    for idx, theta in enumerate(thetas):
        for s in (1, -1):
            ## Update dp when theta change
            if idx > 0 and y_train[idx - 1] > 0:
                dp[idx] = list(map(add, dp[idx - 1], [1, 0]))
            elif idx > 0 and y_train[idx - 1] < 0:
                dp[idx] = list(map(add, dp[idx - 1], [0, 1]))

            ## Calculate err by number of error of (left) and (right)
            if s > 0:
                err = dp[idx][0] + (total_minus_1 - dp[idx][1])
            else:
                err = dp[idx][1] + (total_1 - dp[idx][0])

            ## Get the smallest Ein
            if err / len(y_train) < Ein:
                Ein = err / len(y_train)
                g_theta = theta
                g_s = s
    # print('Ein: {}, g_theta: {}, g_s: {}'.format(Ein, g_theta, g_s))

    return Ein, g_theta, g_s


# ------------------------------------------------------------------------
#                                Problems
# ------------------------------------------------------------------------

@print_problem_number
def Problem(trainSize, tau, Parameters):
    '''
        Generate data and get ans of each problem.

        Parameters:
        trainSize (int): size of training data
        tau (int): flip by probability tau
        Parameters (dict): common parameters

        Returns:
        -
    '''

    ans = []
    for _ in range(Parameters['Epochs']):
        # Generate data
        x_train, y_train = generate_data(trainSize)
        x_test, y_test = generate_data(Parameters['testSize'])

        # Noise
        flip(y_train, tau)
        flip(y_test, tau)

        # Train and get Ein
        Ein, g_theta, g_s = decision_stump(x_train, y_train)

        # Calculate Eout
        Eout = Get_Eout(x_test, y_test, g_theta, g_s)
        # print('Eout:', Eout)

        ans.append(Eout - Ein)

    print(sum(ans) / len(ans))


# ------------------------------------------------------------------------
#                                   Main
# ------------------------------------------------------------------------

if __name__ == '__main__':
    

    # Parameters setting
    Parameters = {
        'Epochs' : 10000,
        'testSize' : 10000,
    }

    # Problem 16 to 20
    Problem(2, 0, Parameters)
    Problem(20, 0, Parameters)
    Problem(2, 0.1, Parameters)
    Problem(20, 0.1, Parameters)
    Problem(200, 0.1, Parameters)