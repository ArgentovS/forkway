from multiprocess import Pool
import pandas as pd
import numpy as np
import datetime as dt


def funct(df):
    df['второй'] = df['первый'] * 100 - df['первый'] / 100000


def parallelize_dataframe(df, func):
    a,b,c = np.array_split(df, 3)
    pool = Pool(8)
    some_res = pool.map(func, [a,b,c])
    df = pd.concat(some_res)
    pool.close()
    pool.join()
    return df


a = dt.datetime.now()
print(a)
df__ = pd.DataFrame({'первый': [elem for elem in range(10000000)]})
#funct(df__)
parallelize_dataframe(df__, funct)
b = dt.datetime.now()
print(b)
print(b -a)