import numpy as np
import math
from qiskit.quantum_info import random_clifford
import scipy as sp
import torch as th
from opt_einsum import contract
import time
import multiprocessing as mp
import matplotlib as mpl
from scipy.optimize import leastsq

mpl.use("Agg")
import matplotlib.pyplot as plt

hmdict = {}
th.cuda.set_device(0)
randomU1 = th.load("./data/random1").cuda()
randomU2 = th.load("./data/random2").cuda()
time_start = time.time()
proc_lst = []
has_activate = np.zeros(11)


def hamming_distance(
    a, b, n, lenth, scheme
):  # the hamming distance between digit n and n+m-1, shceme is either global of local
    x = ((a ^ b) >> n) % (1 << lenth)
    if scheme == "g":
        return int(x != 0)
    else:
        setBits = 0
        while x > 0:
            setBits += x & 1
            x >>= 1
        return setBits


def hamming_distance_table(num_qubit, n, lenth, scheme):
    key = (num_qubit, n, lenth, scheme)
    if key in hmdict:
        return hmdict[key]
    else:
        dim = 2 ** num_qubit
        if scheme == "g":
            hm1 = (-(2 ** lenth)) ** th.tensor(
                [
                    [-hamming_distance(k, l, n, lenth, scheme) for k in range(dim)]
                    for l in range(dim)
                ],
                dtype=th.cfloat,
            )
        else:
            hm1 = (-2) ** th.tensor(
                [
                    [-hamming_distance(k, l, n, lenth, scheme) for k in range(dim)]
                    for l in range(dim)
                ],
                dtype=th.cfloat,
            )
    hmdict[key] = hm1
    return hm1


def GHZ_state(n):
    col = [0, 0, 2 ** n - 1, 2 ** n - 1]
    row = [0, 2 ** n - 1, 0, 2 ** n - 1]
    data = [1 / 2, 1 / 2, 1 / 2, 1 / 2]
    return th.tensor(
        sp.sparse.coo_matrix((data, (row, col)), shape=(2 ** n, 2 ** n)).toarray(),
        dtype=th.cfloat,
    )


def W_state(n):
    col = []
    row = []
    data = []
    for i in range(n):
        for j in range(n):
            col.append(2 ** i)
            row.append(2 ** j)
            data.append(1 / n)
    return th.tensor(
        sp.sparse.coo_matrix((data, (row, col)), shape=(2 ** n, 2 ** n)).toarray(),
        dtype=th.cfloat,
    )


def noisy_GHZ(n, p):  # (1-p)|GHZ><GHZ|+pI/2**n
    ghz = GHZ_state(n) * (1 - p)
    white_noise = th.eye(2 ** n, dtype=th.cfloat) * p / 2 ** n
    return ghz + white_noise


def noisy_W(n, p):  # (1-p)|W><W|+pI/2**n
    w = W_state(n) * (1 - p)
    white_noise = th.eye(2 ** n, dtype=th.cfloat) * p / 2 ** n
    return w + white_noise


def maximally_entangled_state(d):
    psi = [[0 for i in range(d ** 2)]]
    for i in range(d):
        psi[0][i * d + i] = 1 / np.sqrt(d)
    return th.tensor(np.dot(np.conj(psi).T, psi), dtype=th.cfloat)


def realignment(rho, na):
    da = 2 ** na
    db = len(rho) // da
    rho = th.reshape(rho, (da, db, da, db))
    rho = th.swapaxes(rho, 1, 2)  # switch index
    rho = th.reshape(rho, (da * da, db * db))
    return rho


def shadow_estimator(rho, device, scheme, m, qU):
    dim = len(rho)
    qubit_num = int(math.log(len(rho), 2))
    if scheme == "g":
        U_lst = get_u(qubit_num, qU, m).to(device)
    else:
        U_lst_local = get_u(1, None, m * qubit_num).view(qubit_num, m, 2, 2).to(device)
        U_lst = U_lst_local[0]
        for i in range(1, qubit_num):
            U_lst = contract("abc,ade->abdce", U_lst_local[i], U_lst).view(
                m, 2 ** (i + 1), 2 ** (i + 1)
            )
    prob_pdf = contract("oab,bc,oac->oa", U_lst, rho, th.conj(U_lst)).real
    meas_lst = th.empty([m, dim, dim])
    prob_cdf = prob_pdf.cumsum(1)
    randchoice = th.cuda.FloatTensor(m, 1).uniform_()
    result = (randchoice < prob_cdf).type(th.int8).argmax(axis=1)
    meas_range = th.arange(0, m).to(device)
    ones_range = th.ones(m).to(device)
    if scheme == "g":
        meas_lst = (
            th.sparse_coo_tensor(
                th.stack([meas_range, result, result]),
                ones_range,
                (m, dim, dim),
            )
            .to(device)
            .to_dense()
            .type(th.cfloat)
        )
        rho_hat_lst = (dim + 1) * contract(
            "oba,obc,ocd->oad", th.conj(U_lst), meas_lst, U_lst
        ) - th.eye(dim, dtype=th.cfloat).to(device)
    else:
        meas_lst0 = (
            th.sparse_coo_tensor(
                th.stack([meas_range, result % 2, result % 2]),
                ones_range,
                (m, 2, 2),
            )
            .to(device)
            .to_dense()
            .type(th.cfloat)
        )
        rho_hat_lst = 3 * contract(
            "oba,obc,ocd->oad", th.conj(U_lst_local[0]), meas_lst0, U_lst_local[0]
        ) - th.eye(2, dtype=th.cfloat).to(device)
        for i in range(1, qubit_num):
            result=result//2
            meas_lsti = (
                th.sparse_coo_tensor(
                    th.stack([meas_range, result  % 2, result % 2]),
                    ones_range,
                    (m, 2, 2),
                )
                .to(device)
                .to_dense()
                .type(th.cfloat)
            )
            rho_hat_lsti = 3 * contract(
                "oba,obc,ocd->oad", th.conj(U_lst_local[i]), meas_lsti, U_lst_local[i]
            ) - th.eye(2, dtype=th.cfloat).to(device)
            rho_hat_lst = contract("abc,ade->abdce", rho_hat_lsti, rho_hat_lst).view(
                m, 2 ** (i + 1), 2 ** (i + 1)
            )
    return rho_hat_lst


def gen_u(n, qU):
    while True:
        qU.put(random_clifford(n).to_matrix())


def get_u(n, qU, num):
    if n == 2:
        randchoice = th.cuda.FloatTensor(num).uniform_() * 11240
        return randomU2[randchoice.type(th.int64)]
    if n == 1:
        randchoice = th.cuda.FloatTensor(num).uniform_() * 24
        return randomU1[randchoice.type(th.int64)]
    else:
        if has_activate[n]:
            U_lst = th.empty(num, 2 ** n, 2 ** n, dtype=th.cfloat)
            for i in range(num):
                U_lst[i] = th.tensor(qU[n].get(), dtype=th.cfloat)
        else:
            has_activate[n] = 1
            for i in range(15):
                proc = mp.Process(target=gen_u, args=(n, qU[n]))  # Must assign n
                proc_lst.append(proc)
                proc.start()
            U_lst = th.empty(num, 2 ** n, 2 ** n, dtype=th.cfloat)
            for i in range(num):
                U_lst[i] = th.tensor(qU[n].get(), dtype=th.cfloat)
    return U_lst


def terminate():
    print("time is:" + str(time.time() - time_start))
    for proc in proc_lst:
        proc.terminate()


def func(p, x):
    k, b = p
    return k * x + b


def error(p, x, y):
    return func(p, x) - y


def draw(y, x):
    y_log = np.asarray([math.log(var, 2) for var in y])
    plt.figure()
    x_log = np.asarray([math.log(u, 2) for u in x])
    p0 = [1, 1]
    print(leastsq(error, p0, args=(x_log, y_log)))
    plt.scatter(x_log, y_log)
    plt.plot(x_log, y_log, label=" ")
    plt.xlabel(" ")
    plt.ylabel(" ")
    plt.legend()
    plt.savefig("test.png")
def draw_n(y, x):
    y_log = np.asarray([math.log(var, 2) for var in y])
    plt.figure()
    p0 = [1, 1]
    print(leastsq(error, p0, args=(x, y_log)))
    plt.scatter(x, y_log)
    plt.plot(x, y_log, label=" ")
    plt.xlabel(" ")
    plt.ylabel(" ")
    plt.legend()
    plt.savefig("test.png")
def draw_nolog(y, x):
    y_log = np.asarray(y)
    plt.figure()
    p0 = [1, 1]
    print(leastsq(error, p0, args=(x, y_log)))
    plt.scatter(x, y_log)
    plt.plot(x, y_log, label=" ")
    plt.xlabel(" ")
    plt.ylabel(" ")
    plt.legend()
    plt.savefig("test.png")