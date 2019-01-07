#Importing packages

#scikit-learn
from sklearn import preprocessing, decomposition, svm, cross_validation
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

#Visualisation and Manipulation packages
import numpy as np
import matplotlib.pylab as pl
import pandas as pd

#Others
import random

#Supported classifications
classification_supported = {
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'DT': DecisionTreeClassifier()
        'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
    }


class Predictor:
    def __init__(self, dataSet, dependentVar, doFeatureSelection=True, doPCA=False, nComponents=10):
        # Making features compatible with scikit-learn
        for i, tp in enumerate(dataSet.dtypes):
            if tp == 'object':
                print 'Features encoded \"' + dataSet.columns[i] + '\" ...'
                print 'Shape of the old dataset: ' + str(dataSet.shape)
                temp = pd.get_dummies(dataSet[dataSet.columns[i]], prefix=dataSet.columns[i])
                dataSet = pd.concat([dataSet, temp], axis=1).drop(dataSet.columns[i], axis=1)
                print 'Shape of the new dataset: ' + str(dataSet.shape)

        # Appropriate column mapping for the dependent variable
        y = dataSet.loc[:, dependentVar]
        # Eliminate y from the set to be trained upon
        X = dataSet.drop(dependentVar, 1).values

        labels = preprocessing.LabelEncoder().fit_transform(y)

        # Feature engineering
        if doFeatureSelection:
            print 'Before feature engineering: ' + str(X.shape)
            clf = DecisionTreeClassifier(criterion='entropy')
            X = clf.fit(X, y).transform(X)
            print 'After feature engineering ' + str(X.shape) + '\n'

        X = preprocessing.StandardScaler().fit(X).transform(X)

        # Principle Component Analysis for collapsing features
        
        if doPCA:
            estimator = decomposition.PCA(n_components=nComponents)
            X = estimator.fit_transform(X)
            print 'Effect of PCA on the shape: ' + str(X.shape) + '\n'

        # Persisting processed results
        self.dataset = X
        self.labels = labels
        self.students = dataSet.index

    def sub_sampled_data(self, x, y, ix, subsample_ratio=1.0):
        indexes_0 = [item for item in ix if y[item] == 0]
        indexes_1 = [item for item in ix if y[item] == 1]

        sample_length = int(len(indexes_1) * subsample_ratio)
        sample_indexes = random.sample(indexes_0, sample_length) + indexes_1

        return sample_indexes

    #############################################################
    # The MIT License (MIT)                                     #
    # Copyright (c) 2012-2013 Karsten Jeschkies <jeskar@web.de> #
    # By - Karsten Jeschkies                                    #
    #############################################################

    def SMOTE(self, T, N, k, h=1.0):
        n_minority_samples, n_features = T.shape

        if N < 100:
            N = 100
            pass

        if (N % 100) != 0:
            raise ValueError("N must be either greater than 100 or a multiple of 100")

        N = N / 100
        n_synthetic_samples = N * n_minority_samples
        S = np.zeros(shape=(n_synthetic_samples, n_features))

        # Nearest neighbor learner
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(T)

        # Synthetic sample calculations
        for i in xrange(n_minority_samples):
            nn = neigh.kneighbors(T[i], return_distance=False)
            for n in xrange(N):
                nn_index = random.choice(nn[0])
                
                while nn_index == i:
                    nn_index = random.choice(nn[0])

                dif = T[nn_index] - T[i]
                gap = np.random.uniform(low=0.0, high=h)
                S[n + i * N, :] = T[i, :] + gap * dif[:]

        return S

    def classifier_learner(self, outputFormat='score', doSubsampling=False, subRate=1.0,
                          doSMOTE=False, pctSMOTE=100, nFolds=10, models=['LR'], topK=.1, ):

        # Computation of accuracy score
        if outputFormat == 'score':
            if doSMOTE or doSubsampling:
                print 'Sorry, scoring with subsampling or SMOTE not yet implemented'
                return
            # Evaluating classifier by iteration
            for ix, clf in enumerate([classification_supported[x] for x in models]):
                kf = cross_validation.KFold(len(self.dataset), nFolds, shuffle=True)
                scores = cross_validation.cross_val_score(clf, self.dataset, self.labels, cv=kf)
                print models[ix] + ' Accuracy of the model: %.2f' % np.mean(scores)

        # Confusion Matrix
        elif outputFormat == 'summary' or outputFormat == 'matrix':
            for ix, clf in enumerate([classification_supported[x] for x in models]):
                y_prediction_results = [];
                y_smote_prediction_results = []
                y_original_values = []

                # K-fold index generation
                kf = cross_validation.StratifiedKFold(self.labels, n_folds=nFolds)
                for i, (train, test) in enumerate(kf):
                    if doSubsampling:
                        train = self.sub_sampled_data(self.dataset, self.labels, train, subRate)
                    if doSMOTE:
                        minority = self.dataset[train][np.where(self.labels[train] == 1)]
                        smotted = self.SMOTE(minority, pctSMOTE, 5)
                        X_train_smote = np.vstack((self.dataset[train], smotted))
                        y_train_smote = np.append(self.labels[train], np.ones(len(smotted), dtype=np.int32))
                        y_pred_smote = clf.fit(X_train_smote, y_train_smote).predict(self.dataset[test])
                        y_smote_prediction_results = np.concatenate((y_smote_prediction_results, y_pred_smote), axis=0)

                    fitted_clf = clf.fit(self.dataset[train], self.labels[train])
                    y_pred = fitted_clf.predict(self.dataset[test])
                    y_prediction_results = np.concatenate((y_prediction_results, y_pred), axis=0)
                    y_original_values = np.concatenate((y_original_values, self.labels[test]), axis=0)

                # K-fold summary table
                if outputFormat == 'summary':
                    print '\t' + models[ix] + ' Summary Results'
                    confusion_matrix = classification_report(y_original_values, y_prediction_results, target_names=['Did Graduate', 'Dropped Out'])
                    print(str(confusion_matrix) + '\n')
                    if doSMOTE:
                        print '\t' + models[ix] + ' SMOTE Summary Results'
                        confusion_matrix = classification_report(y_original_values, y_smote_prediction_results, target_names=['Did Graduate', 'Dropped Out'])
                        print(str(confusion_matrix) + '\n')

                else:
                    print '\t' + models[ix] + ' Confusion Matrix'
                    print '\t Did Graduate \t Dropped Out'
                    confusion_matrix = confusion_matrix(y_original_values, y_prediction_results)
                    print 'Did Graduate\t\t\t%d\t\t%d' % (confusion_matrix[0][0], confusion_matrix[0][1])
                    print 'Dropped Out\t%d\t\t%d' % (confusion_matrix[1][0], confusion_matrix[1][1])
                    if doSMOTE:
                        print '\n' + models[ix] + ' SMOTE Confusion Matrix'
                        print 'Did Graduate \t Dropped Out'
                        confusion_matrix = confusion_matrix(y_original_values, y_smote_prediction_results)
                        print 'Did Graduate\t %d\t %d' % (confusion_matrix[0][0], confusion_matrix[0][1])
                        print 'Dropped Out\t%d\t%d' % (confusion_matrix[1][0], confusion_matrix[1][1])

        #ROC Curve Computation
        elif outputFormat == 'roc':
            for ix, clf in enumerate([classification_supported[x] for x in models]):
                kf = cross_validation.StratifiedKFold(self.labels, n_folds=nFolds)
                mean_tpr = mean_smote_tpr = 0.0
                mean_fpr = mean_smote_fpr = np.linspace(0, 1, 100)

                for i, (train, test) in enumerate(kf):
                    if doSubsampling:
                        train = self.sub_sampled_data(self.dataset, self.labels, train, subRate)
                    if doSMOTE:
                        minority = self.dataset[train][np.where(self.labels[train] == 1)]
                        smotted = self.SMOTE(minority, pctSMOTE, 5)
                        X_train = np.vstack((self.dataset[train], smotted))
                        y_train = np.append(self.labels[train], np.ones(len(smotted), dtype=np.int32))
                        probas2_ = clf.fit(X_train, y_train).predict_proba(self.dataset[test])
                        fpr, tpr, thresholds = roc_curve(self.labels[test], probas2_[:, 1])
                        mean_smote_tpr += np.interp(mean_smote_fpr, fpr, tpr)
                        mean_smote_tpr[0] = 0.0

                    # Probability for the predicted samples
                    fitted_clf = clf.fit(self.dataset[train], self.labels[train])
                    probas_ = fitted_clf.predict_proba(self.dataset[test])

                    # ROC & Area of the Curve generation
                    fpr, tpr, thresholds = roc_curve(self.labels[test], probas_[:, 1])
                    mean_tpr += np.interp(mean_fpr, fpr, tpr)

                # ROC Baseline Plot
                pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Baseline')

                # Computation of the true positives
                mean_tpr /= len(kf)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)

                pl.plot(mean_fpr, mean_tpr, 'k-',
                        label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

                # Oversampling Plots
                if doSMOTE:
                    mean_smote_tpr /= len(kf)
                    mean_smote_tpr[-1] = 1.0
                    mean_smote_auc = auc(mean_smote_fpr, mean_smote_tpr)
                    pl.plot(mean_smote_fpr, mean_smote_tpr, 'r-',
                            label='Mean smote ROC (area = %0.2f)' % mean_smote_auc, lw=2)

                pl.xlim([-0.05, 1.05])
                pl.ylim([-0.05, 1.05])
                pl.xlabel('False Positive Rate')
                pl.ylabel('True Positive Rate')
                pl.title(models[ix] + ' ROC')
                pl.legend(loc="lower right")
                pl.show()

        elif outputFormat == 'prc' or outputFormat == 'topk' or outputFormat == 'risk':
            for ix, clf in enumerate([classification_supported[x] for x in models]):
                y_prob = [];
                y_smote_prob = []
                y_prediction_results = [];
                y_smote_prediction_results = []
                y_original_values = [];
                test_indexes = []

                kf = cross_validation.StratifiedKFold(self.labels, n_folds=nFolds, shuffle=True)

                for i, (train, test) in enumerate(kf):
                    if doSubsampling:
                        train = self.sub_sampled_data(self.dataset, self.labels, train, subRate)

                    if doSMOTE:
                        clf2 = clf
                        minority = self.dataset[train][np.where(self.labels[train] == 1)]
                        smotted = self.SMOTE(minority, pctSMOTE, 5)
                        X_train = np.vstack((self.dataset[train], smotted))
                        y_train = np.append(self.labels[train], np.ones(len(smotted), dtype=np.int32))
                        clf2.fit(X_train, y_train)
                        probas2_ = clf2.predict_proba(self.dataset[test])
                        y_pred_smote = clf2.predict(self.dataset[test])
                        y_smote_prediction_results = np.concatenate((y_smote_prediction_results, y_pred_smote), axis=0)
                        y_smote_prob = np.concatenate((y_smote_prob, probas2_[:, 1]), axis=0)

                    clf.fit(self.dataset[train], self.labels[train])
                    y_pred = clf.predict(self.dataset[test])
                    y_prediction_results = np.concatenate((y_prediction_results, y_pred), axis=0)
                    test_indexes = np.concatenate((test_indexes, test), axis=0)
                    y_original_values = np.concatenate((y_original_values, self.labels[test]), axis=0)
                    probas_ = clf.predict_proba(self.dataset[test])
                    y_prob = np.concatenate((y_prob, probas_[:, 1]), axis=0)

                # Compute overall prediction, recall and area under PR-curve
                precision, recall, thresholds = precision_recall_curve(y_original_values, y_prob)
                pr_auc = auc(recall, precision)

                if doSMOTE:
                    precision_smote, recall_smote, thresholds_smote = precision_recall_curve(y_original_values,
                                                                                             y_smote_prob)
                    pr_auc_smote = auc(recall_smote, precision_smote)

                # Output the precision recall curve
                if outputFormat == 'prc':
                    pl.plot(recall, precision, color='b', label='Precision-Recall curve (area = %0.2f)' % pr_auc)
                    if doSMOTE:
                        pl.plot(recall_smote, precision_smote, color='r',
                                label='SMOTE Precision-Recall curve (area = %0.2f)' % pr_auc_smote)
                    pl.xlim([-0.05, 1.05])
                    pl.ylim([-0.05, 1.05])
                    pl.xlabel('Recall')
                    pl.ylabel('Precision')
                    pl.title(models[ix] + ' Precision-Recall')
                    pl.legend(loc="lower right")
                    pl.show()

                # Output a list of the topK% students at highest risk along with their risk scores
                elif outputFormat == 'risk':
                    test_indexes = test_indexes.astype(int)
                    sort_ix = np.argsort(test_indexes)
                    students_by_risk = self.students[test_indexes]
                    y_prob = ((y_prob[sort_ix]) * 100).astype(int)
                    probas = np.column_stack((students_by_risk, y_prob))
                    r = int(topK * len(y_original_values))
                    print models[ix] + ' top ' + str(100 * topK) + '%' + ' highest risk'
                    print '%-15s %-10s' % ('Student', 'Risk Score')
                    print '%-15s %-10s' % ('-------', '----------')
                    probas = probas[np.argsort(probas[:, 1])[::-1]]
                    for i in range(r):
                        print '%-15s %-10d' % (probas[i][0], probas[i][1])
                    print '\n'

                # Understanding the prediction of Top K %
                else:
                    ord_prob = np.argsort(y_prob, )[::-1]
                    r = int(topK * len(y_original_values))
                    print models[ix] + ' Precision for top ' + str(100 * topK) + '%'
                    print np.sum(y_original_values[ord_prob][:r]) / r

                    if doSMOTE:
                        ord_prob = np.argsort(y_smote_prob, )[::-1]
                        print models[ix] + ' SMOTE Precision for top ' + str(100 * topK) + '%'
                        print np.sum(y_original_values[ord_prob][:r]) / r
                    print '\n'