""" Preprocessing Data for Decision Tree, Bagged Trees and Random Forest"""
import pandas as pd
import numpy as np

random_state = 47
frac = 0.2

preference_scores_of_participant = ['attractive_important', 'sincere_important', 'intelligence_important', \
    'funny_important', 'ambition_important', 'shared_interests_important']
preference_scores_of_partner = ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', \
'pref_o_ambitious', 'pref_o_shared_interests']
not_continuous_valued_columns = ['gender', 'samerace', 'decision']

def preprocess(input_file):
    df = pd.read_csv(str(input_file))

    # use only 6500 rows
    df = df.head(min(6500, df.shape[0]))
    # print(df.shape)

    # 1(i) Drop race, race_o and field columns
    df.drop(columns = ['race', 'race_o', 'field'], inplace=True)

    #1(ii) Converting gender attribute to label encoding
    df['gender'] = df['gender'].astype("category")
    df['gender'] = df['gender'].cat.codes

    #1(iii) Normalization
    preference_scores_of_participant_sum = df[preference_scores_of_participant].sum(axis = 1)
    preference_scores_of_partner_sum = df[preference_scores_of_partner].sum(axis = 1)
    df[preference_scores_of_participant] = df[preference_scores_of_participant].div(preference_scores_of_participant_sum, axis = 0)
    df[preference_scores_of_partner] = df[preference_scores_of_partner].div(preference_scores_of_partner_sum, axis = 0)

    #1(iv) Discretization
    default_scale_min = 0
    default_scale_max = 10
    preference_scores_scale_min = 0
    preference_scores_scale_max = 1
    correlation_columns = ['interests_correlate']
    correlation_scale_scale_min = -1
    correlation_scale_scale_max = 1
    age_columns = ['age', 'age_o']
    age_scale_min = 18
    age_scale_max = 58
    lower_value, upper_value = 0, 10

    for column in df.columns:
        if column in not_continuous_valued_columns:
            continue
        elif column in preference_scores_of_participant or column in preference_scores_of_partner:
            lower_value = preference_scores_scale_min
            upper_value = preference_scores_scale_max
        elif column in correlation_columns:
            lower_value = correlation_scale_scale_min
            upper_value = correlation_scale_scale_max
        elif column in age_columns:
            lower_value = age_scale_min
            upper_value = age_scale_max
        else:
            lower_value = default_scale_min
            upper_value = default_scale_max
        df[column].clip(lower = lower_value, upper = upper_value, inplace = True)
        bins = np.linspace(lower_value, upper_value, 3)
        df[column] = pd.cut(x = df[column], bins = bins, labels = [0, 1], include_lowest = True)

    #1(v) random sampling
    test_df = df.sample(frac = frac, random_state = random_state)
    train_df = df.drop(test_df.index)

    return train_df, test_df

if __name__ == '__main__':
    input_file = 'dating-full.csv'
    train_df, test_df = preprocess(input_file)
    train_df.to_csv('trainingSet.csv', index=False)
    test_df.to_csv('testSet.csv', index=False)
