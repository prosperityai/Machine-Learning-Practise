import classifier
import pandas as pd

# Converting the CSV input file into a datframe
df = pd.read_csv('../../data/dropout_data.csv',index_col=0)

# Classification Model
pred = classifier.Predictor(df,'nograd')

# Logistic Regression on the classification model using 10-fold cross validation
pred.classifier_learner(outputFormat='risk', models=['LR'], nFolds=10)