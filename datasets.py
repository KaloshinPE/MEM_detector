import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler




def one_hot_df(df, df_cols, scaler=None, test=False):
    
    df_1 = df.drop(columns = df_cols)
    if scaler:
        if test:
            df_1 = pd.DataFrame(scaler.transform(df_1.values), columns=df_1.columns)
        else:
            df_1 = pd.DataFrame(scaler.fit_transform(df_1.values), columns=df_1.columns)
        
    df_2 = pd.get_dummies(df[list(df_cols)])
    
    return (pd.concat([df_1, df_2], axis=1))


def get_adults_dataframes(data_dir='data'):
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num',\
                'marital-status', 'occupation', 'relationship', 'race',\
                'gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country','income']
    df_train = pd.read_csv(f'{data_dir}/adult/adult.data.txt', header=None, names=column_names)
    df_test = pd.read_csv(f'{data_dir}/adult/adult.test.txt', skiprows=1, names=column_names)
    
    scaler = StandardScaler()
    df_train = one_hot_df(df_train, df_train.select_dtypes('object'), scaler)\
                                .drop(columns=['income_ <=50K'])
    target = lambda x: 1 if x else -1

    df_train['target']=df_train['income_ >50K'].apply(target)
    df_train = df_train.drop(columns = ['income_ >50K', 'native-country_ Holand-Netherlands'])
    
    df_test = one_hot_df(df_test, df_test.select_dtypes('object'), scaler, test=True)\
                                    .drop(columns=['income_ <=50K.'])
    df_test['target']=df_test['income_ >50K.'].apply(target)
    df_test = df_test.drop(columns = ['income_ >50K.'])
    
    return df_train, df_test

def split_dataframe(df):
    return df.drop(columns=['target']).values, df['target'].values[:,None]

def get_train_test(df_train, df_test):
    X_train, y_train = split_dataframe(df_train)
    X_test, y_test = split_dataframe(df_test)
    
    return X_train, y_train, X_test, y_test

def get_dataset(name):
    if name=='adults':
        df_train, df_test = get_adults_dataframes()
        X_train, y_train, X_test, y_test = get_train_test(df_train, df_test)
    return X_train, y_train, X_test, y_test