""" Implementation of DT, Bagged Trees and Random Forests"""
import sys

import pandas as pd
import numpy as np
import math
import time
import pprint

from multiprocessing import Pool 

multiprocessing_ = True
num_processes = 6
min_examples = 50
label = 'decision'

np.random.seed(0)

def accuracy(original_labels, predicted_labels):
    count = 0
    total_num = len(original_labels)
    for idx in range(total_num):
        if original_labels[idx] == predicted_labels[idx]:
            count += 1
    return float(count)/total_num

##############################################################

def gini(target_column):
    _, counts = np.unique(target_column, return_counts = True)
    return 1 - sum(pow(count/len(target_column), 2) for count in counts)

def giniGain(train_df, attribute, label):
    total_gini = gini(train_df[label])

    elements, counts = np.unique(train_df[attribute], return_counts=True)
    weighted_gini = sum(counts[i]/len(train_df) * gini(train_df[train_df[attribute] == elements[i]][label]) for i in range(len(elements)))

    return total_gini - weighted_gini

def decisionTreeHelper(train_df, attributes, attributes_used, max_depth, sample_attrs = False):
    one_count = train_df[label].sum()
    zero_count = len(train_df) - one_count
    most_freq_label = 1 if one_count > zero_count else 0

    # stop growing a tree if we meet with any of these criteria
    if len(train_df) < min_examples or max_depth == 0:
        return most_freq_label
    elif one_count == len(train_df):
        return 1
    elif zero_count == len(train_df):
        return 0
    else:
        attributes_v = attributes
        if sample_attrs == True:
            # sampling sqrt(p) attributes
            n = len(attributes) # n = 49, total number of attributes
            samples = np.random.choice(n-1, int(math.sqrt(n)), replace=False)
            attributes_v = [attributes[idx] for idx in samples]

        # Select the attribute with the most information gain
        attribute_points = [giniGain(train_df, attribute, label) if attribute in attributes_v \
                                and attribute not in attributes_used else -1 for attribute in attributes]
        best_attribute_index = np.argmax(attribute_points)
        best_attribute = attributes[best_attribute_index]

        # create a decision tree
        tree = {best_attribute:{}}

        # add the best attribute to the list of used attributes
        attributes_used.append(best_attribute)
        tree[best_attribute]['*'] = most_freq_label

        for attr_val in train_df[best_attribute].unique():
            train_df_v = train_df[train_df[best_attribute] == attr_val]
            tree[best_attribute][attr_val] = decisionTreeHelper(train_df_v, attributes, list(attributes_used), max_depth - 1, sample_attrs)

    return tree

def decisionTree(train_df, test_df, max_depth = 8):
    # print(train_df.describe())

    attributes = train_df.drop(columns=label).columns
    tree = decisionTreeHelper(train_df, attributes, [], max_depth)
    # pprint.pprint(tree)

    train_accuracy = accuracy(list(train_df.iloc[:,-1]), predictDT(train_df, tree))
    test_accuracy = accuracy(list(test_df.iloc[:,-1]), predictDT(test_df, tree))

    return train_accuracy, test_accuracy

def predictDT(input_df, tree):
    result = []
    for idx in range(len(input_df)):
        predicted_result = predictQuery(input_df.iloc[idx,:], tree)
        result.append(predicted_result)
    return result

def predictQuery(query, tree):
    tree_key, tree_value = list(tree.items())[0]
    attrs = list(tree_value.keys())
    if query[tree_key] in attrs:
        result = tree[tree_key][query[tree_key]]
    else:
        result = tree[tree_key]['*']
    # recursively call predict on subtrees until you reach the leaf node
    if isinstance(result, dict):
        result = predictQuery(query, result)
    return result

##############################################################

def bagging(train_df, test_df, max_depth = 8, num_trees = 30):
    # s = time.time()
    attributes = train_df.drop(columns=label).columns

    trees = []
    if multiprocessing_:
        results_objs = []
        with Pool(processes=num_processes) as p:
            for _ in range(num_trees):
                # bootstrapping the data
                train_df_v = train_df.sample(frac = 1, replace=True)
                tree_mp = p.apply_async(decisionTreeHelper, (train_df_v, attributes, [], max_depth))
                results_objs.append(tree_mp)
            trees = [result.get() for result in results_objs]
    else:
        for _ in range(num_trees):
            train_df_v = train_df.sample(frac = 1, replace=True)
            trees.append(decisionTreeHelper(train_df_v, attributes, [], max_depth))

    # predictions
    train_accuracy = accuracy(list(train_df.iloc[:,-1]), predictBT(train_df, trees))
    test_accuracy = accuracy(list(test_df.iloc[:,-1]), predictBT(test_df, trees))

    return train_accuracy, test_accuracy

def predictBT(input_df, trees):
    result = []

    for idx in range(len(input_df)):
        query_result = []
        for tree in trees:
            predicted_result = predictQuery(input_df.iloc[idx,:], tree)
            query_result.append(predicted_result)
        result.append(1 if sum(query_result)/len(query_result) >= 0.5 else 0)
    
    return result

#####################################################################

def randomForests(train_df, test_df, max_depth = 8, num_trees = 30):
    attributes = train_df.drop(columns=label).columns    
    trees = []
    if multiprocessing_:
        results_objs = []
        with Pool(processes=num_processes) as p:
            for _ in range(num_trees):
                # bootstrapping the data
                train_df_v = train_df.sample(frac = 1, replace=True)
                tree_mp = p.apply_async(decisionTreeHelper, (train_df_v, attributes, [], max_depth, True))
                results_objs.append(tree_mp)
            trees = [result.get() for result in results_objs]
    else:
        for _ in range(num_trees):
            # bootstrapping the data
            train_df_v = train_df.sample(frac = 1, replace=True)
            trees.append(decisionTreeHelper(train_df_v, attributes, [], max_depth, sample_attrs = True))

    # predictions
    train_accuracy = accuracy(list(train_df.iloc[:,-1]), predictRF(train_df, trees))
    test_accuracy = accuracy(list(test_df.iloc[:,-1]), predictRF(test_df, trees))

    return train_accuracy, test_accuracy

def predictRF(input_df, trees):
    result = []

    for idx in range(len(input_df)):
        query_result = []
        for tree in trees:
            predicted_result = predictQuery(input_df.iloc[idx,:], tree)
            query_result.append(predicted_result)
        result.append(1 if sum(query_result)/len(query_result) >= 0.5 else 0)

    return result

######################################################################

def perform(train_df, test_df, model_idx):
    model = ''
    if model_idx == 1:
        model, (train_acc, test_acc) = 'DT', decisionTree(train_df, test_df)
    elif model_idx == 2:
        model, (train_acc, test_acc) = 'BT', bagging(train_df, test_df)
    else:
        model, (train_acc, test_acc) = 'RF', randomForests(train_df, test_df)
    print(f'Training Accuracy {model}: {round(train_acc,2)}')
    print(f'Testing Accuracy {model}: {round(test_acc,2)}')

if __name__ == '__main__':
    st = time.time()
    training_data_filename = sys.argv[1]
    test_data_file_name = sys.argv[2]
    model_idx =  int(sys.argv[3])
    train_df = pd.read_csv(str(training_data_filename))
    test_df = pd.read_csv(str(test_data_file_name))
    perform(train_df, test_df, model_idx)
    print(f'Total Time Elapsed: {time.time()-st} seconds')
