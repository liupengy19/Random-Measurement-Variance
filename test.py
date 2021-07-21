import torch as th
import numpy as np
from opt_einsum import contract
from qiskit.quantum_info import random_clifford
import cProfile
import pstats
from pstats import SortKey
import time
# th.cuda.set_device(2)
# time1=time.time()
# y=th.FloatTensor(10000, 100000).normal_()
# print(y.size())
# print(str(time.time()-time1))
x=th.tensor([False,True,False,False],dtype=th.bool).type(th.int8)

print(th.argmax(x))