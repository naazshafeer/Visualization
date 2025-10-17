import numpy
import pandas as pd
import os


#this will be the cross check python script to see if the WISE .tbl and the csv file SDSS designations match.

#This python script will compare the WISE .tbl and SDSS dataframe CSV and then output them ones that don't match (also referring to the index)

#The ones that dont match with will be run again to see if there are other sources (so in the WISE catalog without the one-to-one match)

def load_tbl(file_tbl):
    with open(file_tbl, 'r') as t_in:
        