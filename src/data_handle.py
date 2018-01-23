#!/usr/bin/env python

import sqlite3
import pandas as pd
import numpy as np

"""
A collection of data handling convenience functions for storing experiment
data in a SQLite database.
"""

def add_result(c,dim,start_num,dist,mid_num,fin_num,run_num,
                isd,time,test_mix,result_mix,runnalls_isd,runnalls_time,runnalls_mix):
    """
    Add experiement result with parameters

    - params: dictionary of experiment parameter values
    """
    if np.isnan(isd):
        isd = 99
    if np.isnan(runnalls_isd):
        runnalls_isd = 99

    print('INSERT INTO data VALUES ({},{},{},{},{},{},{},{},{},{})'\
        .format(dim,start_num,dist,mid_num,fin_num,run_num,isd,time,runnalls_isd,runnalls_time))
    test_mix_buf = buffer(test_mix)
    result_mix_buf = buffer(result_mix)
    runnalls_buf = buffer(runnalls_mix)
    # print(type(test_mix_buf))
    c.execute("INSERT INTO data VALUES ({d},{sn},'{dis}',{mn},{fn},{rn},{i},{t},'{tm}','{rm}',{run_i},{t_i},'{run}')"\
        .format(d=dim,sn=start_num,dis=dist,mn=mid_num,fn=fin_num,rn=run_num,i=isd,\
                t=time,tm=test_mix_buf,rm=result_mix_buf,run_i=runnalls_isd,t_i=runnalls_time,run=runnalls_buf,))

def get_result():
    pass

def create_table(c,conn,tbl_name):
    """
    Create a new table in a database
    """
    c.execute('CREATE TABLE {tn} (dim integer, start_num integer, dist text,\
                mid_num integer, fin_num integer, run_num integer, ISD real,\
                time real, test_mix blob, result_mix blob, runnalls_ISD real,\
                runnalls_time real, runnalls_mix blob)'.format(tn=tbl_name))

def connect(db_file):
    """
    Connect to a SQLite database file, return the file handle and cursor object

    Returns:
    - conn: database file handle
    - c: cursor object for manipulating the database
    """
    try:
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
    except sqlite3.DatabaseError as e:
        c, conn = None
        print(e)

    return c, conn

def close(c,conn,commit=True):
    """
    Commit changes to database and close connection to database file
    """
    if commit:
        conn.commit()
    c.close()
    conn.close()

def load_data(filename,tablename):
    """
    Load all data from test results sqlite database using sql query to pandas
    dataframe.

    Params:
        - filename: string, database filename
        - tablename: string, name of table in database from which to pull data

    Returns:
        - data: pandas dataframe containing all data
    """
    conn = sqlite3.connect(filename)
    data = pd.read_sql_query("SELECT * FROM '{}'".format(tablename),conn)
    conn.close()
    return data

def grab_data(data,dims=None,sn=None,mn=None,fn=None,dist=None):
    """
    Grab data with specified parameters from tests results dataframe

    Params:
        - data: pandas data frame containg all test data (returned from fxn
                load_data())
        - dim: desired dimension of data
        - sn: desired starting number of mixands
        - mn: desired middle number of clusters
        - fn: desired final number of mixands per cluster
        - dist: desired distance metric or metrics
    """
    params_df = pd.DataFrame(columns=['dim','start_num','dist','mid_num','fin_num',
                                        'run_num','ISD','time','test_mix','result_mix',
                                        'runnalls_ISD','runnalls_time','runnalls_mix'])

    # make sure passed parameters are lists to iterate through,
    # if passed param is None, iterate through all params
    if dims is None:
        dims = [1,2,4]
    elif dims is not list:
        dims = [dims]

    if sn is None:
        sn = [100,400,700,1000]
    elif dims is not list:
        sn = [sn]

    if mn is None:
        mn = [4,10,15]
    elif mn is not list:
        mn = [mn]
    
    if fn is None:
        fn = [5,10,25]
    elif fn is not list:
        fn = [fn]

    if dist is None:
        dist = ['symKLD','JSD','euclid','EMD','bhatt']
    elif dist is not list:
        dist = [dist]

    # iterate through param lists and append grabbed data
    for dim in dims:
        for start_num in sn:
            for mid_num in mn:
                for fin_num in fn:
                    for distance in dist:

                        new_df = data[(data['dim'] == dim) & (data['start_num'] == start_num) &
                                    (data['mid_num'] == mid_num) & (data['fin_num'] == fin_num) &
                                    (data['dist'] == distance)]
                        params_df = params_df.append(new_df)
                        del new_df
    return params_df


if __name__ == "__main__":
    fn = 'test_db.sqlite'
    conn,c = connect(fn)
    fields = ['dim','start_num','dist','mid']
    create_table(c,conn,'new_table','dim')
    close(c,conn)


