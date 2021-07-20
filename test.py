import torch as th
import numpy as np
from opt_einsum import contract
from qiskit.quantum_info import random_clifford
import cProfile
import pstats
from pstats import SortKey
mydict={}
i=0
mylist=th.empty([24,2,2],dtype=th.cfloat)
while(i<24):
    x=random_clifford(1)
    key=str(x.to_dict())
    if key not in mydict:
        mylist[i]=th.tensor(x.to_matrix(),dtype=th.cfloat)
        i+=1
        mydict[key]=1
th.save(mylist, 'random1')
