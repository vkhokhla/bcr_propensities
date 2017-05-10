import pandas as pd
import numpy as np
import timeit

def preprocess(df):
    df[df['AGE'] > 100 * 12] = -1 
    df[df['TENOR'] > 30 * 12] = -1 
    return df
    

df_1612 = pd.read_pickle('../cache/c_201612.pkl')
preprocess(df_1612)