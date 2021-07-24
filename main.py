from M4 import *

qU = [mp.Queue(100) for i in range(11)]
device = th.device("cuda:0")
#! Naming convention: run_(lr/gr/ls/gs)_(state)_(parameter+range_range)
#! plot3
def run_lr_W4_Nu2_Minf_Na2_1024():
    reslist = np.zeros(17)
    for i in range(17):
        reslist[i] += random_m4_variance(
            noisy_W(4, 0),
            na=2,
            NuA=2,
            NuB=2,
            Nm=-1,
            N_iid=200,
            device=device,
            scheme="l",
            qU=qU,
            N_average=int(2 * 2.0 ** (i / 2.0)),
        )
    np.save("./output/lr_W4_Nu2_Minf_Na2_1024_iid200", reslist)
    draw_n(reslist, range(17))


def run_gr_W4_Nu2_Minf_Na2_1024():
    reslist = np.zeros(17)
    for i in range(17):
        reslist[i] += random_m4_variance(
            noisy_W(4, 0),
            na=2,
            NuA=2,
            NuB=2,
            Nm=-1,
            N_iid=200,
            device=device,
            scheme="g",
            qU=qU,
            N_average=int(2 * 2.0 ** (i / 2.0)),
        )
    np.save("./output/gr_W4_Nu2_Minf_Na2_1024_iid200", reslist)
    draw_n(reslist, range(17))


def run_lr_W4_Nu2_M5_Na2_1024():
    reslist = np.zeros(17)
    for i in range(17):
        reslist[i] += random_m4_variance(
            noisy_W(4, 0),
            na=2,
            NuA=2,
            NuB=2,
            Nm=5,
            N_iid=200,
            device=device,
            scheme="l",
            qU=qU,
            N_average=int(2 * 2.0 ** (i / 2.0)),
        )
    np.save("./output/lr_W4_Nu2_M5_Na2_1024_iid200", reslist)
    draw_n(reslist, range(17))


def run_gr_W4_Nu2_M5_Na2_1024():
    reslist = np.zeros(17)
    for i in range(17):
        reslist[i] += random_m4_variance(
            noisy_W(4, 0),
            na=2,
            NuA=2,
            NuB=2,
            Nm=5,
            N_iid=200,
            device=device,
            scheme="g",
            qU=qU,
            N_average=int(2.0 ** (i / 2.0)),
        )
    np.save("./output/gr_W4_Nu2_M5_Na2_1024_iid200", reslist)
    draw_n(reslist, range(17))


#! plot4
def run_gr_W4_Nu2_M1_512_inf_Na10():
    reslist = np.zeros(18)
    for i in range(17):
        reslist[i] += random_m4_variance(
            noisy_W(4, 0),
            na=2,
            NuA=2,
            NuB=2,
            Nm=int(2.0 ** (i / 2.0)),
            N_iid=200,
            device=device,
            scheme="g",
            qU=qU,
            N_average=10,
        )
    reslist[17] += random_m4_variance(
        noisy_W(4, 0),
        na=2,
        NuA=2,
        NuB=2,
        Nm=-1,
        N_iid=2000,
        device=device,
        scheme="g",
        qU=qU,
        N_average=10,
    )
    np.save("./output/gr_W4_Nu2_M1_512_inf_Na10_iid2000", reslist)
    draw_n(reslist, range(18))


def run_lr_W4_Nu2_M1_512_inf_Na10():
    reslist = np.zeros(18)
    for i in range(17):
        reslist[i] += random_m4_variance(
            noisy_W(4, 0),
            na=2,
            NuA=2,
            NuB=2,
            Nm=int(2.0 ** (i / 2.0)),
            N_iid=200,
            device=device,
            scheme="l",
            qU=qU,
            N_average=10,
        )
    reslist[17] += random_m4_variance(
        noisy_W(4, 0),
        na=2,
        NuA=2,
        NuB=2,
        Nm=-1,
        N_iid=2000,
        device=device,
        scheme="l",
        qU=qU,
        N_average=10,
    )
    np.save("./output/lr_W4_Nu2_M1_512_inf_Na10_iid2000", reslist)
    draw_n(reslist, range(18))


def run_gr_W4_Nu2_M1_512_inf_Na20():
    reslist = np.zeros(18)
    for i in range(17):
        reslist[i] += random_m4_variance(
            noisy_W(4, 0),
            na=2,
            NuA=2,
            NuB=2,
            Nm=int(2.0 ** (i / 2.0)),
            N_iid=200,
            device=device,
            scheme="g",
            qU=qU,
            N_average=20,
        )
    reslist[17] += random_m4_variance(
        noisy_W(4, 0),
        na=2,
        NuA=2,
        NuB=2,
        Nm=-1,
        N_iid=2000,
        device=device,
        scheme="g",
        qU=qU,
        N_average=20,
    )
    np.save("./output/gr_W4_Nu2_M1_512_inf_Na20_iid2000", reslist)
    draw_n(reslist, range(18))


def run_lr_W4_Nu2_M1_512_inf_Na20():
    reslist = np.zeros(18)
    for i in range(17):
        reslist[i] += random_m4_variance(
            noisy_W(4, 0),
            na=2,
            NuA=2,
            NuB=2,
            Nm=int(2.0 ** (i / 2.0)),
            N_iid=200,
            device=device,
            scheme="l",
            qU=qU,
            N_average=20,
        )
    reslist[17] += random_m4_variance(
        noisy_W(4, 0),
        na=2,
        NuA=2,
        NuB=2,
        Nm=-1,
        N_iid=2000,
        device=device,
        scheme="l",
        qU=qU,
        N_average=20,
    )
    np.save("./output/lr_W4_Nu2_M1_512_inf_Na20_iid2000", reslist)
    draw_n(reslist, range(18))


#! plot1
def run_lr_W2_10_Nu2_M4_Na5():
    reslist = np.zeros(9)
    for n in range(2, 11):
        reslist[n - 2] += random_m4_variance(
            noisy_W(n, 0),
            na=n // 2,
            NuA=2,
            NuB=2,
            Nm=4,
            N_iid=20,
            device=device,
            scheme="l",
            qU=qU,
            N_average=5,
        )
    np.save("./output/lr_W2_10_Nu2_M4_Na5_iid20", reslist)
    draw_n(reslist, range(9))


def run_gr_W2_10_Nu2_M4_Na5():
    reslist = np.zeros(9)
    for n in range(2, 11):
        reslist[n - 2] += (
            random_m4_variance(
                noisy_W(n, 0),
                na=n // 2,
                NuA=2,
                NuB=2,
                Nm=4,
                N_iid=20,
                device=device,
                scheme="g",
                qU=qU,
                N_average=5,
            )
            / 20
        )
    np.save("./output/gr_W2_10_Nu2_M4_Na5_iid20", reslist)
    draw_n(reslist, range(9))


def run_ls_W2_10_M80_Na1():
    reslist = np.zeros(9)
    for time in range(20):
        print(time)
        for n in range(2, 11):
            reslist[n - 2] += (
                shadow_m4_variance(
                    noisy_W(n, 0),
                    na=n // 2,
                    m=4,
                    N_iid=1,
                    device=device,
                    scheme="l",
                    qU=qU,
                )
                / 20
            )
    np.save("./output/ls_W2_10_M80_Na1_iid20", reslist)
    draw_n(reslist, range(9))


def run_gs_W2_10_M80_Na1():
    reslist = np.zeros(9)
    for time in range(20):
        print(time)
        for n in range(2, 11):
            reslist[n - 2] += (
                shadow_m4_variance(
                    noisy_W(n, 0),
                    na=n // 2,
                    m=80,
                    N_iid=1,
                    device=device,
                    scheme="g",
                    qU=qU,
                )
                / 20
            )
    np.save("./output/gs_W2_10_M80_Na1_iid20", reslist)
    draw_n(reslist, range(9))


#! plot2
def run_gs_W4_M4_256_Na1():
    reslist = np.zeros(13)
    for n in range(13):
        reslist[n] += shadow_m4_variance(
            noisy_W(4, 0),
            na=2,
            m=int(4 * 2.0 ** (n / 2.0)),
            N_iid=200,
            device=device,
            scheme="g",
            qU=qU,
        )
    np.save("./output/gs_W4_M4_256_Na1_iid200", reslist)
    draw_n(reslist, range(13))


def run_ls_W4_M4_256_Na1():
    reslist = np.zeros(13)
    for n in range(13):
        reslist[n] += shadow_m4_variance(
            noisy_W(4, 0),
            na=2,
            m=int(4 * 2.0 ** (n / 2.0)),
            N_iid=200,
            device=device,
            scheme="l",
            qU=qU,
        )
    np.save("./output/ls_W4_M4_256_Na1_iid200", reslist)
    draw_n(reslist, range(13))


if __name__ == "__main__":
    run_gr_W4_Nu2_M5_Na2_1024()
    terminate()
