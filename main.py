from M4 import *

qU = [mp.Queue(100) for i in range(11)]
reslist = np.zeros(99)
device = th.device("cuda:0")
#! Naming convention: run_(lr/gr/ls/gs)_(state)_(parameter+range_range)
#! plot3
def run_lr_W4_Nu2_Minf_Na1_2500():
    for i in range(40):
        print(i)
        for u in range(2, 101):
            reslist[u - 2] += (
                random_m4_variance(
                    noisy_W(4, 0),
                    na=2,
                    NuA=2,
                    NuB=2,
                    Nm=-1,
                    N_iid=50,
                    device=device,
                    scheme="l",
                    qU=qU,
                    N_average=u ** 2 // 4,
                )
                / 40
            )
    th.save(th.tensor(reslist), "./output/lr_W4_Nu2_Minf_Na1_2500")


def run_lr_W4_Nu2_100_Minf_Na1():
    for i in range(20):
        print(i)
        for u in range(2, 101):
            reslist[u - 2] += (
                random_m4_variance(
                    noisy_W(4, 0),
                    na=2,
                    NuA=u,
                    NuB=u,
                    Nm=-1,
                    N_iid=100,
                    device=device,
                    scheme="l",
                    qU=qU,
                )
                / 20
            )
    th.save(th.tensor(reslist), "./output/lr_W4_Nu2_100_Minf_Na1")


def run_lr_W4_Nu2_100_M20_Na1():
    for i in range(20):
        print(i)
        for u in range(2, 101):
            reslist[u - 2] += (
                random_m4_variance(
                    noisy_W(4, 0),
                    na=2,
                    NuA=u,
                    NuB=u,
                    Nm=20,
                    N_iid=100,
                    device=device,
                    scheme="l",
                    qU=qU,
                )
                / 20
            )
    th.save(th.tensor(reslist), "./output/lr_W4_Nu2_100_M20_Na1")


def run_gr_W4_Nu2_Minf_Na1_2500():
    for i in range(40):
        print(i)
        for u in range(2, 101):
            reslist[u - 2] += (
                random_m4_variance(
                    noisy_W(4, 0),
                    na=2,
                    NuA=2,
                    NuB=2,
                    Nm=-1,
                    N_iid=50,
                    device=device,
                    scheme="g",
                    qU=qU,
                    N_average=u ** 2 // 4,
                )
                / 40
            )
    th.save(th.tensor(reslist), "./output/gr_W4_Nu2_Minf_Na1_2500")


def run_gr_W4_Nu2_100_Minf_Na1():
    for i in range(20):
        print(i)
        for u in range(2, 101):
            reslist[u - 2] += (
                random_m4_variance(
                    noisy_W(4, 0),
                    na=2,
                    NuA=u,
                    NuB=u,
                    Nm=-1,
                    N_iid=100,
                    device=device,
                    scheme="g",
                    qU=qU,
                )
                / 20
            )
    th.save(th.tensor(reslist), "./output/gr_W4_Nu2_100_Minf_Na1")


def run_gr_W4_Nu2_100_M20_Na1():
    for i in range(20):
        print(i)
        for u in range(2, 101):
            reslist[u - 2] += (
                random_m4_variance(
                    noisy_W(4, 0),
                    na=2,
                    NuA=u,
                    NuB=u,
                    Nm=20,
                    N_iid=100,
                    device=device,
                    scheme="g",
                    qU=qU,
                )
                / 20
            )
    th.save(th.tensor(reslist), "./output/gr_W4_Nu2_100_M20_Na1")


#! plot4
def run_gr_W4_Nu10_M1_100_Na1():
    for m in range(1, 100):
        reslist[m - 1] += random_m4_variance(
            noisy_W(4, 0),
            na=2,
            NuA=10,
            NuB=10,
            Nm=m,
            N_iid=3000,
            device=device,
            scheme="g",
            qU=qU,
        )
    th.save(th.tensor(reslist), "./output/gr_W4_Nu10_M1_100_Na1")


def run_lr_W4_Nu10_M1_100_Na1():
    for m in range(1, 100):
        reslist[m - 1] += random_m4_variance(
            noisy_W(4, 0),
            na=2,
            NuA=10,
            NuB=10,
            Nm=m,
            N_iid=3000,
            device=device,
            scheme="l",
            qU=qU,
        )
    th.save(th.tensor(reslist), "./output/lr_W4_Nu10_M1_100_Na1")


def run_gr_W4_Nu20_M1_100_Na1():
    for m in range(1, 100):
        reslist[m - 1] += random_m4_variance(
            noisy_W(4, 0),
            na=2,
            NuA=20,
            NuB=20,
            Nm=m,
            N_iid=2000,
            device=device,
            scheme="g",
            qU=qU,
        )
    th.save(th.tensor(reslist), "./output/gr_W4_Nu20_M1_100_Na1")


def run_lr_W4_Nu20_M1_100_Na1():
    for m in range(1, 100):
        reslist[m - 1] += random_m4_variance(
            noisy_W(4, 0),
            na=2,
            NuA=20,
            NuB=20,
            Nm=m,
            N_iid=2000,
            device=device,
            scheme="l",
            qU=qU,
        )
    th.save(th.tensor(reslist), "./output/lr_W4_Nu20_M1_100_Na1")


reslist = np.zeros(9)
#! plot1
def run_lr_W2_9_Nu3_M10_Na1():
    for time in range(1000):
        for n in range(2, 10):
            reslist[n - 2] += (
                random_m4_variance(
                    noisy_W(n, 0),
                    na=n // 2,
                    NuA=3,
                    NuB=3,
                    Nm=10,
                    N_iid=1,
                    device=device,
                    scheme="l",
                    qU=qU,
                )
                / 1000
            )
    th.save(th.tensor(reslist), "./output/lr_W2_9_Nu3_M10_Na1")


def run_gr_W2_9_Nu3_M10_Na1():
    for time in range(1000):
        for n in range(2, 10):
            reslist[n - 2] += (
                random_m4_variance(
                    noisy_W(n, 0),
                    na=n // 2,
                    NuA=3,
                    NuB=3,
                    Nm=10,
                    N_iid=1,
                    device=device,
                    scheme="g",
                    qU=qU,
                )
                / 1000
            )
    th.save(th.tensor(reslist), "./output/gr_W2_9_Nu3_M10_Na1")


def run_ls_W2_9_M60_Na1():
    for time in range(100):
        for n in range(2, 10):
            reslist[n - 2] += (
                shadow_m4_variance(
                    noisy_W(n, 0),
                    na=n // 2,
                    m=60,
                    N_iid=1,
                    device=device,
                    scheme="l",
                    qU=qU,
                )
                / 100
            )
    th.save(th.tensor(reslist), "./output/ls_W2_9_M60_Na1")


def run_gs_W2_9_M60_Na1():
    for time in range(10):
        for n in range(2, 10):
            reslist[n - 2] += (
                shadow_m4_variance(
                    noisy_W(n, 0),
                    na=n // 2,
                    m=60,
                    N_iid=1,
                    device=device,
                    scheme="g",
                    qU=qU,
                )
                / 10
            )
    th.save(th.tensor(reslist), "./output/gs_W2_9_M60_Na1")


#! plot2
def run_gs_W4_M10_90_Na1():
    for n in range(10, 100, 10):
        reslist[n // 10 - 1] += shadow_m4_variance(
            noisy_W(4, 0), na=2, m=n, N_iid=200, device=device, scheme="g", qU=qU
        )
    th.save(th.tensor(reslist), "./output/gs_W4_M10_90_Na1")


def run_ls_W4_M10_90_Na1():
    for n in range(10, 100, 10):
        reslist[n // 10 - 1] += shadow_m4_variance(
            noisy_W(4, 0), na=2, m=n, N_iid=500, device=device, scheme="l", qU=qU
        )
    th.save(th.tensor(reslist), "./output/ls_W4_M10_90_Na1")


if __name__ == "__main__":
    run_gs_W4_M10_90_Na1()
    terminate()
    draw(reslist, range(1, 10))
