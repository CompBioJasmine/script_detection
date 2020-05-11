#!/usr/bin/python
import numpy as np
import random
import argparse
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             roc_curve, roc_auc_score)

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import sklearn
#import matplotlib.pyplot as plt
#import pandas
import math
import operator
import csv

def find_optimal_cutoff_closest_to_perfect(fpr, tpr, threshold):
  """ Tries to find the best threshold to select on ROC.

  """
  distance = float("inf")
  threshold_index = -1
  for i in range(len(fpr)):
    d1 = math.sqrt((fpr[i] - 0) * (fpr[i] - 0) + (tpr[i] - 1) * (tpr[i] - 1))
    if d1 < distance:
      distance = d1
      threshold_index = i

  return threshold_index

def analysis(figurenum,
             train_y,
             train_X,
             test_y,
             test_X):
  """
    Performs the analysis.

    Arguments:
    figurenum - Used to increment the figurenum between calls so we don't plot
                different data to the same figure.
    train_y - The training labels.
    train_X - The training data.
    test_y - The test labels.
    test_X - The test data.
    plot - If true, plots the ROC and Precision/Recall curves.
    problem_name - Used to label the plot.
    test_src_ips - The source ips for the netflows of test set
    test_dest_ips - The dest ips for the netflows of the test set
  """

  clf = RandomForestClassifier()
  clf.fit(train_X, train_y)

  train_scores = clf.predict_proba(train_X)[:,1]
  test_scores = clf.predict_proba(test_X)[:,1]
  train_auc = roc_auc_score(train_y, train_scores)
  test_auc = roc_auc_score(test_y, test_scores)
  train_average_precision = average_precision_score(train_y, train_scores)
  test_average_precision = average_precision_score(test_y, test_scores)
  fpr, tpr, thresholds = roc_curve(test_y, test_scores)
  threshold_index = find_optimal_cutoff_closest_to_perfect(fpr, tpr, thresholds)

  print( "AUC of ROC on train", train_auc )
  print( "AUC of ROC on test", test_auc )
  print( "Average Precision on train", train_average_precision )
  print( "Average Precision on test", test_average_precision )
  print( "Optimal point", fpr[threshold_index], tpr[threshold_index])


def random_split(items1, items2):
  print(items1.shape)
  print(items2.shape)
  a1, a2, b1, b2 = np.array([[]]), np.array([[]]), [],[]
  for i in range(len(items1)):
    if np.random.random() < 0.5:
      b1.append(items2[i])
      a1 = np.append(a1, [items1[i]])
    else:
      a2 = np.append(a2, [items1[i]])
      b2.append(items2[i])

  return a1, a2, b1, b2


def main():

  # Process command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--inputfiletest', type=str, required=True,
                      help="The test with the features and labels.")
  parser.add_argument('--inputfiletrain', type=str, required=True,
                      help="The train with the features and labels.")

  parser.add_argument('--subset', type=str,
    help="Comma-separated list of features to include")

  FLAGS = parser.parse_args()

  # Open a file with the extracted features
  with open(FLAGS.inputfiletest, "r") as test, open(FLAGS.inputfiletrain, "r") as train:
    datate = np.loadtxt(train, delimiter=",")
    y1 = datate[:, 0] #labels are in the first column
    X1 = datate[:, 1:] #features are in the rest of the columns
    datatr = np.loadtxt(test, delimiter=",")
    y2 = datatr[:, 0] #labels are in the first column
    X2 = datatr[:, 1:] #features are in the rest of the columns
    if FLAGS.subset:
      selectedFeatures = FLAGS.subset.split(",")
      selectedFeatures = list(map(int, selectedFeatures))
      X1 = datate[:, selectedFeatures]
      print( X1[1] )
      X2 = datatr[:, selectedFeatures]
      print( X2[1] )

    #y2 = np.asarray(y2)
    #X2 = np.asarray(X2)
    #print("Test file is", str(file))
    print( "Length y1, y2", len(y1), len(y2) )
    print( "Share X1, X2", X1.shape, X2.shape )
    print( "Nonzero y1, y2", np.count_nonzero(y1), np.count_nonzero(y2) )
    figurenum = 0
    figurenum = analysis(figurenum, y1, X1, y2, X2)#, srcIps[i:], destIps[i:])
    figurenum = analysis(figurenum, y2, X2, y1, X1)#, srcIps[0:i], destIps[0:i])


main()
