"""Measuring Influence of Tree Depth on Classifier Performance"""
import math
import time
import pandas as pd

from cv_plot import plot_curves
from trees import bagging, decisionTree, randomForests

def crossValidationDepth(train_df, model_idx):
    accuracy_list = []
    std_error_list = []

    sample_size = int(train_df.shape[0]/10)
    S = [train_df.iloc[i*sample_size:(i+1)*sample_size,:] for i in range(10)]

    for max_depth in depth_limit:
        accuracy_cv = []
        for idx in range(10):
            test_set = S[idx]
            list_ = []
            for item in range(10):
                not item == idx and list_.append(item)
            train_set = pd.concat(S[item] for item in list_)
            if model_idx == 1:
                _, test_acc = decisionTree(train_set, test_set, max_depth)
            elif model_idx == 2:
                _, test_acc = bagging(train_set, test_set, max_depth)
            else:
                _, test_acc = randomForests(train_set, test_set, max_depth)
            accuracy_cv.append(test_acc)
            # print('fold: ', idx, 'test_acc: ', test_acc)
        avg_accuracy = sum(accuracy_cv)/len(accuracy_cv)
        print('max_depth: ', max_depth, 'avg_accuracy: ', avg_accuracy)
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
    depth_limit = [3, 5, 7, 9]
    # cross validation depth for DT
    print('###########################################')
    print('Cross Validation for DT')
    acc_list1, err_list1 = crossValidationDepth(train_df, 1)
    # cross validation depth for BT
    print('###########################################')
    print('Cross Validation for BT')
    acc_list2, err_list2 = crossValidationDepth(train_df, 2)
    # cross validation depth for RF
    print('###########################################')
    print('Cross Validation for RF')
    acc_list3, err_list3 = crossValidationDepth(train_df, 3)
    # plotting the curves
    print('###########################################')
    print(f'Total Time Elapsed: {round(time.time()-s,2)} seconds')
    plot_curves(depth_limit, 'Depth of Tree', acc_list1, err_list1, acc_list2, err_list2, acc_list3, err_list3)