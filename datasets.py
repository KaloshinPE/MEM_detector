import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



SEED = 2

def one_hot_df(df, scaler=None, test=False, discrete=False):
    df_cols = df.select_dtypes('object')
    df_1 = df.drop(columns = df_cols).drop(columns = ['target'])
    if scaler:
        if test:
            df_1 = pd.DataFrame(scaler.transform(df_1.values))
        else:
            df_1 = pd.DataFrame(scaler.fit_transform(df_1.values))
    df_2 = pd.get_dummies(df[list(df_cols)])
    
    if discrete:
        return pd.concat([df_2, df[['target']]], axis=1)  
    
    return  (pd.concat([df_1, df_2, df[['target']]], axis=1))

def get_hepatitis_dataframes(discrete=False):
    df = pd.read_csv('data/hepatitis.data', header=None).rename(columns = {0:'target'})
    df.target = df.target.apply(lambda x: 1 if x==1 else -1)
    df=one_hot_df(df, discrete=discrete)
    df_train, df_test = train_test_split(df, random_state=SEED)
    return df_train, df_test

def get_votes_dataframes():
    df = pd.read_csv('data/house-votes-84.data', header=None).rename(columns = {0:'target'})
    df.target = df.target.apply(lambda x: 1 if x=='democrat' else -1)
    df=one_hot_df(df)
    df_train, df_test = train_test_split(df, random_state=SEED)
    return df_train, df_test

def get_kr_kp_dataframes():
    df = pd.read_csv('data/kr-vs-kp.data', header=None).rename(columns = {36:'target'})
    df.target = df.target.apply(lambda x: 1 if x=='won' else -1)
    df=one_hot_df(df)
    df_train, df_test = train_test_split(df, random_state=SEED)
    return df_train, df_test

def get_prometers_dataframes(discrete=False):
    df = pd.read_csv('data/promoters.data', header=None).drop(columns=[0]).rename(columns = {58:'target'})
    df.target = df.target.apply(lambda x: 1 if x=='+' else -1)
    df = one_hot_df(df, discrete=discrete)
    df_train, df_test = train_test_split(df, random_state=SEED)
    return df_train, df_test

def get_credits_dataframes(discrete=False):
    df = pd.read_csv('data/crx.data', header=None).rename(columns = {15:'target'})
    df[1]=df[1].replace('?', '0').apply(eval)
    df.target = df.target.apply(lambda x: 1 if x=='+' else -1)
    df=one_hot_df(df, discrete=discrete)
    df_train, df_test = train_test_split(df, random_state=SEED)
    return df_train, df_test

def get_adults_dataframes(discrete=False):
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num',\
                'marital-status', 'occupation', 'relationship', 'race',\
                'gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country','income']
    df_train = pd.read_csv('data/adult.data', header=None).rename(columns = {14:'target'})
    df_train.target = df_train.target.apply(lambda x: 1 if x==' >50K' else -1)
    
    df_test = pd.read_csv('data/adult.test', skiprows=1, header=None).rename(columns = {14:'target'})
    df_test.target = df_test.target.apply(lambda x: 1 if x==' >50K.' else -1)
    
    scaler = StandardScaler()
    
    df_train = one_hot_df(df_train, scaler, discrete=discrete)
    df_train = df_train.drop(columns = ['13_ Holand-Netherlands'])
    
    df_test = one_hot_df(df_test, scaler, test=True, discrete=discrete)
    
    return df_train, df_test

def split_dataframe(df):
    return df.drop(columns=['target']).values, df['target'][:, None]

def get_train_test(df_train, df_test):
    X_train, y_train = split_dataframe(df_train)
    X_test, y_test = split_dataframe(df_test)
    
    return X_train, y_train, X_test, y_test

def get_dataset(name, discrete=False):
    if name=='adult':
        df_train, df_test = get_adults_dataframes(discrete=discrete)        
    elif name=='credits':
        df_train, df_test = get_credits_dataframes(discrete=discrete)
    elif name=='kr-vs-kp':
        df_train, df_test = get_kr_kp_dataframes()
    elif name=='promoters':
        df_train, df_test = get_prometers_dataframes()
    elif name=='votes':
        df_train, df_test = get_votes_dataframes()
    elif name=='hepatitis':
        df_train, df_test = get_hepatitis_dataframes(discrete=discrete)
        
    X_train, y_train, X_test, y_test = get_train_test(df_train, df_test)
    return X_train, y_train, X_test, y_test