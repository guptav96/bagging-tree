"""Measuring Influence of Number of Trees on Classifier Performance"""
import math
import time
import pandas as pd

from cv_plot import plot_curves
from trees import bagging, decisionTree, randomForests

def crossValidationTrees(train_df, model_idx):
    accuracy_list = []
    std_error_list = []

    sample_size = int(train_df.shape[0]/10)
    S = [train_df.iloc[i*sample_size:(i+1)*sample_size,:] for i in range(10)]

    for num_trees in tree_list:
        accuracy_cv = []
        for idx in range(10):
            test_set = S[idx]
            list_ = []
            for item in range(10):
                not item == idx and list_.append(item)
            train_set = pd.concat(S[item] for item in list_)
            if model_idx == 2:
                _, test_acc = bagging(train_set, test_set, num_trees = num_trees)
            else:
                _, test_acc = randomForests(train_set, test_set, num_trees = num_trees)
            accuracy_cv.append(test_acc)
            # print('fold: ', idx, 'test_acc: ', test_acc)
        avg_accuracy = sum(accuracy_cv)/len(accuracy_cv)
        print('num_trees: ', num_trees, 'avg_accuracy: ', avg_accuracy)
        variance = sum([((x - avg_accuracy) ** 2) for x in accuracy_cv]) / len(accuracy_cv)
        std_error = (variance ** 0.5) / math.sqrt(10)
        accuracy_list.append(avg_accuracy)
        std_error_list.append(std_error)

    return accuracy_list, std_error_list


if __name__ == '__main__':
    s = time.time()
    train_df = pd.read_csv('trainingSet.csv')
    train_df = train_df.sample(frac = 1, random_state = 18)
    train_df = train_df.sample(frac = 0.5, random_state = 32)
    tree_list = [10, 20, 40, 50]
    # cross validation trees for BT
    print('###########################################')
    print('Cross Validation for BT')
    acc_list2, err_list2 = crossValidationTrees(train_df, 2)
    # cross validation trees for RF
    print('###########################################')
    print('Cross Validation for RF')
    acc_list3, err_list3 = crossValidationTrees(train_df, 3)
    print('###########################################')
    # plotting the curves
    print(f'Time Elapsed: {time.time()-s} seconds')
    plot_curves(tree_list, 'Number of Trees', acc_list2, err_list2, acc_list3, err_list3)  