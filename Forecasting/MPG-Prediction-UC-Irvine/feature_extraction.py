#!/usr/bin/env python

import warnings
import sys
from multiprocessing import mProc
import numpy as np
from sklearn import support_vec_machine as support_vec_machine
import pandas AS pd
import math
import pylab

warnings.filterwarnings("ignore", category=FutureWarning, append=1)

CV_SETS = 5
INTERESTING_FEATURES = ('cyls', 'displacement', 'hp', 'weight', 'acc', 'year', 'origin')


def extract_lables_and_features(csv_file):
    """Extract labels and features from relevant part of the CSV data"""
    df = pd.read_csv(csv_file)
    labels = df['mpg'].values
    features = zip(*map(lambda f: df[f].values, INTERESTING_FEATURES))
    return features, labels


def cross_validation_sets(feature_vectors, labels, n=CV_SETS):
    """Constructing n-sets of labels and features for cross-validation."""
    feature_sets = [ [] for i in range(n) ]
    label_sets = [ [] for i in range(n) ]
    feature_it = iter(feature_vectors)
    label_it = iter(labels)
    i = 0
    try:
        while True:
            feature_sets[i].append(feature_it.next())
            label_sets[i].append(label_it.next())
            i = (i + 1) % n
    except:
        pass
    return feature_sets, label_sets


def index_rotate(index):
    return index.insert(0, index.pop())


def permutation(feature_sets, label_sets, index):
    """Preparing train and test sets from list"""
    train_features, train_labels = [], []
    for i in index[1:]:
        train_features += feature_sets[i]
        train_labels += label_sets[i]
    return train_features, train_labels, feature_sets[index[0]], label_sets[index[0]]


def score_parameters(feature_sets, label_sets, C, gamma):
    """Run all possible permutations for cross-validation of the given support_vec_machine parameters."""
    index = range(CV_SETS)
    totals = [0] * CV_SETS
    for perm in range(CV_SETS):
        train_f, train_l, test_f, test_l = permutation(feature_sets, label_sets, index)
        clf = support_vec_machine.SVR(C=C, gamma=gamma, kernel='rbf', scale_C=False)
        clf.fit(train_f, train_l)
        for i in range(len(test_f)):
            f, l = test_f[i], test_l[i]
            totals[perm] += (l - clf.predict(f)[0]) ** 2
        index_rotate(index)

    return C, gamma, sum(totals) / CV_SETS


def test_parameters(feature_sets, label_sets, C, gamma):
    """Run cross-validation on with the given support_vec_machine parameters using the process mProc."""
    return proc_mProc.apply(score_parameters, (feature_sets, label_sets, C, gamma))


def param_permutations(C_min, C_max, C_no, gamma_min, gamma_max, gamma_no):
    """Within the given window, finding out the param tuples"""
    Cs = np.linspace(C_min, C_max, num=C_no)
    gammas = np.linspace(gamma_min, gamma_max, num=gamma_no)
    out = []
    for c in Cs:
        for gamma in gammas:
            out.append( (c, gamma) )
    return out


def _score_parameters_map(args):
    return score_parameters(*args)


def test_param_permuations(feature_sets, label_sets, permuations):
    aug_permuations = map(lambda x: tuple([feature_sets, label_sets] + list(x)), permuations)
    return proc_mProc.map(_score_parameters_map, aug_permuations)


def hone(C_min, C_max, C_no, gamma_min, gamma_max, gamma_no, feature_sets, label_sets,
         new_window_proportion=0.5, epsilon=0.1, last_error=0, max_iterations=10):

    # Computation of best minimum parameter
    permuations = param_permutations(C_min, C_max, C_no, gamma_min, gamma_max, gamma_no)
    results = test_param_permuations(feature_sets, label_sets, permuations)
    C_best, gamma_best, error_best = sorted(results, key=lambda x: x[2])[0]

    print >> sys.stderr
    print >> sys.stderr, "Window: C:", (C_min, C_max), "  gamma:", (gamma_min, gamma_max)
    print >> sys.stderr, "Best C:", C_best, "  gamma:", gamma_best, "  error:", error_best

    if (abs(last_error - error_best) <= epsilon) or (max_iterations == 1):
        return C_best, gamma_best, error_best

    # Applying {C, gamma} > 0 for present window size computation
    C_diff = C_max - C_min
    gamma_diff = gamma_max - gamma_min
    C_offset = C_diff * new_window_proportion * 0.5
    gamma_offset = gamma_diff * new_window_proportion * 0.5

    C_min_new = C_best - C_offset
    C_max_new = C_best + C_offset
    if C_min_new < 0:
        C_max_new += abs(C_min_new)
        C_min_new = 1

    gamma_min_new = gamma_best - gamma_offset
    gamma_max_new = gamma_best + gamma_offset
    if gamma_min_new < 0:
        gamma_max_new += abs(gamma_min_new)
        gamma_min_new = 1e-20

    # recursive step
    return hone(C_min_new, C_max_new, C_no, gamma_min_new, gamma_max_new, gamma_no,
                feature_sets, label_sets, new_window_proportion=new_window_proportion,
                epsilon=epsilon, last_error=error_best, max_iterations=max_iterations-1)


def std_dev(Xs):
    """Standard deviation of the items in a list."""
    mean = sum(Xs) / len(Xs)
    return math.sqrt(sum(map(lambda x: (x-mean)**2, Xs))/(len(Xs)-1))

# it appears mProc() needs to be run /after/ the function it'll need to apply
proc_mProc = mProc(processes=None)

if __name__ == '__main__':

    # get best parameters
    feature_sets, label_sets = cross_validation_sets(*extract_lables_and_features("input.csv"))
    C, gamma, error = hone(100, 20000, 10, 1e-9, 1e-5, 10, feature_sets, label_sets,
                           new_window_proportion=0.6, max_iterations=3)

    clf = support_vec_machine.SVR(kernel='rbf', C=C, gamma=gamma, scale_C=False)
    features = reduce(lambda x,y: x+y, feature_sets)
    labels = reduce(lambda x,y: x+y, label_sets)
    clf.fit(features, labels)

    # Error computation
    errors = []
    for feature, label in zip(features, labels):
        errors.append(clf.predict(feature)[0] - label)
    errors = sorted(errors)

    print
    print "Error's SD:", std_dev(errors)

    # construct a crude histogram of errors
    granularity = 30
    bins = [0] * granularity
    _min, _max = errors[0], errors[-1]
    diff = _max - _min
    errors = map(lambda x: x-_min, errors)
    step = diff / (granularity - 1)
    bin_nos = map(lambda x: int(math.floor(x/step)), errors)
    for i in bin_nos:
        bins[i] += 1

    # clear bins
    x, y = [], []
    for a, b in zip(range(granularity), bins):
        if b > 0:
            x.append(_min + a * step)
            y.append(b)

    # plot
    pylab.plot(x, y)
    pylab.show()