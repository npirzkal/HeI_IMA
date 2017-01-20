import glob, sys
import numpy as np


def get_visits(names):
	"""Identified files that can be assumed to be from the same visit. Returns a dictionary grouping the files by visit"""

    if type(names)==type([]):
        files = names
    elif os.path.isfile(names):
        files = open(names).readlines()
    else:
        files = glob.glob(names)
    dic = {}
    for f in files:
        k = f[0:6]
        if not dic.has_key(k):
            dic[k]= []
        dic[k].append(f.strip())
    return dic


if __name__=="__main__":
	raw_names = glob.glob(sys.argv[1])
    print("Processing",raw_names)
    visits = get_visits(raw_names)