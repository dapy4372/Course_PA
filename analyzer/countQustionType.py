import os
import sys
import csv
import numpy as np
import argparse

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default='error')
    parser.add_argument('-dir', type=str, default='./error')
    return parser.parse_args()

if __name__ == "__main__":
    arg = parseArgs()

    if arg.mode == 'error':
        print '*** error mode ***'
        fileList = []
        for file in os.listdir(arg.dir):
            if file.endswith('.csv'):
                fileList.append(file)
    else:
        print '*** all question mode ***'
