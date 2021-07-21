from M4 import *
import multiprocessing as mp
import time
import cProfile
from draw import *

# def runM4var():
if __name__ == "__main__":
    proc_lst = []
    qU = [mp.Queue(100) for i in range(6)]
    # for i in range(10):
    #     proc = mp.Process(target=gen_u, args=(2, qU[2]))  # Must assign n
    #     proc_lst.append(proc)
    #     proc.start()
    time1 = time.time()
    reslist = []
    device = th.device("cuda:0")  # "cpu" to use cpu, cuda:0 to use gpu
    for u in range(2, 100):
        print(u)
        res = random_m4_variance(noisy_W(4, 0), 2, u, u, 1, 20, 100, device, "l", qU)
        # print(res)
        reslist.append(res / 21)
    for proc in proc_lst:
        proc.terminate()
    print(str(time.time() - time1))
    th.save(th.tensor(reslist), "./output/random_l_W(4)_Nu=2to100_M=20")
    draw2(reslist, range(2, 100))
