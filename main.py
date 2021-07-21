from M4 import *

# def runM4var():
if __name__ == "__main__":
    qU = [mp.Queue(100) for i in range(11)]
    reslist = []
    device = th.device("cuda:0")  # "cpu" to use cpu, cuda:0 to use gpu
    print(shadow_m4_variance(noisy_W(2, 0), 1, 100, 100, device, "l", qU))
    # for u in range(2, 50):
    #     print(u)
    #     res = random_m4_variance(noisy_W(4, 0), 2, u, u, 1, 20, 100, device, "g", qU)
    #     # print(res)
    #     reslist.append(res / 21)
    kill()
    th.save(th.tensor(reslist), "./output/random_l_W(4)_Nu=2to100_M=20")
    draw(reslist, range(2, 50))
