#!/usr/bin/env python

import sqlite3
import pandas as pd

"""
A collection of data handling convenience functions for storing experiment
data in a SQLite database.
"""

def add_result(c,dim,start_num,dist,mid_num,fin_num,isd,time):
    """
    Add experiement result with parameters

    - params: dictionary of experiment parameter values
    """
    # print('INSERT INTO data VALUES ({},{},{},{},{},{},{})'\
        # .format(dim,start_num,dist,mid_num,fin_num,isd,time))
    c.execute("INSERT INTO data VALUES ({},{},'{}',{},{},{},{},{},{})"\
        .format(dim,start_num,dist,mid_num,fin_num,isd,time))

def get_result():
    pass

def create_table(c,conn):
    """
    Create a new table in a database
    """
    c.execute('CREATE TABLE data (dim integer, start_num integer, dist text,\
                mid_num integer, fin_num integer, ISD real, time real, \
                test_mix blob, result_mix blob)')

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


