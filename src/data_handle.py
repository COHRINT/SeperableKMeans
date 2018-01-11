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


if __name__ == "__main__":
    fn = 'test_db.sqlite'
    conn,c = connect(fn)
    fields = ['dim','start_num','dist','mid']
    create_table(c,conn,'new_table','dim')
    close(c,conn)


