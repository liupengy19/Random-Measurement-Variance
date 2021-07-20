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
    time1=time.time()
    mylist = []
    device = th.device("cpu")  # "cpu" to use cpu, cuda:0 to use gpu
    for u in range(2, 100):
        print(u)
        res = random_m4_variance(noisy_W(4, 0), 2, u, u, 1, 20, 100, device, "l", qU)
        # print(res)
        mylist.append(res/21)
    # for i in range(20):
    #     print(i)
    #     for u in range(2, 100):
    #         res = random_m4_variance(noisy_W(4, 0), 2, u, u, 1, 20, 1, device, "l", qU)
    #         # print(res)
    #         mylist[u-2]+=res/21
    for proc in proc_lst:
        proc.terminate()
    x=th.tensor(mylist)
    print(str(time.time()-time1))
    th.save(x,"output/random_l_W(4)_Nu=2to100_M=20")
    draw2(mylist, range(2, 100))
    # cProfile.run() profile the perf