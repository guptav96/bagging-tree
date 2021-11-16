"""Plotting Learning Curves for all models: DT, BT and RF"""
import math
import time
import pandas as pd

from cv_plot import plot_curves
from trees import bagging, decisionTree, randomForests

def crossValidationFrac(train_df, model_idx):
    accuracy_list = []
    std_error_list = []

    sample_size = int(train_df.shape[0]/10)
    S = [train_df.iloc[i*sample_size:(i+1)*sample_size,:] for i in range(10)]

    for t_frac in frac_list:
        accuracy_cv = []
        for idx in range(10):
            test_set = S[idx]
            list_ = []
            for item in range(10):
                not item == idx and list_.append(item)
            S_c = pd.concat(S[item] for item in list_)
            train_set = S_c.sample(frac = t_frac, random_state = 32)
            if model_idx == 1:
                _, test_acc = decisionTree(train_set, test_set)
            elif model_idx == 2:
                _, test_acc = bagging(train_set, test_set)
            else:
                _, test_acc = randomForests(train_set, test_set)
            accuracy_cv.append(test_acc)
            # print('fold: ', idx, 'test_acc: ', test_acc)
        avg_accuracy = sum(accuracy_cv)/len(accuracy_cv)
        print('t_frac: ', t_frac, 'avg_accuracy: ', avg_accuracy)
        variance = sum([((x - avg_accuracy) ** 2) for x in accuracy_cv]) / len(accuracy_cv)
        std_error = (variance ** 0.5) / math.sqrt(10)
        accuracy_list.append(avg_accuracy)
        std_error_list.append(std_error)

    return accuracy_list, std_error_list


if __name__ == '__main__':
    s = time.time()
    train_df = pd.read_csv('trainingSet.csv')
    train_df = train_df.sample(frac = 1, random_state = 18)
    frac_list = [0.05, 0.075, 0.1, 0.15, 0.2]
    # cross validation depth for DT
    print('###########################################')
    print('Cross Validation for DT')
    acc_list1, err_list1 = crossValidationFrac(train_df, 1)
    # cross validation depth for BT
    print('###########################################')
    print('Cross Validation for BT')
    acc_list2, err_list2 = crossValidationFrac(train_df, 2)
    # cross validation depth for RF
    print('###########################################')
    print('Cross Validation for RF')
    acc_list3, err_list3 = crossValidationFrac(train_df, 3)
    print('###########################################')
    # plotting the curves
    print(f'Total Time Elapsed: {round(time.time()-s,2)} seconds')
    plot_curves(list(map(lambda x: x * len(train_df), frac_list)), 'Number of training samples', acc_list1, err_list1, acc_list2, err_list2, acc_list3, err_list3)