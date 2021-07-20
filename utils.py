import numpy as np
import math
from qiskit.quantum_info import random_clifford
import scipy as sp
import torch as th
from opt_einsum import contract

hmdict = {}


def hamming_distance(
    a, b, n, lenth, scheme
):  # a,b are numbers, give the hamming distance between digit n and n+m-1, shceme is either global of local
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


def GHZ_state(n):  # generate GHZ state, n stands for qubit number
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
    U_lst = th.empty([m, dim, dim], dtype=th.cfloat)
    if scheme == "g":
        U_lst = get_u(qubit_num,qU,m)
    else:
        U_lst_local = get_u(1,None,m*qubit_num)
        U_lst_local.to(device)
        #!contract
    U_lst.to(device)
    prob_lst = abs(contract("oab,bc,oac->oa", U_lst, rho, th.conj(U_lst)).real)
    prob_lst_norm = prob_lst.sum(1) ** (-1)
    prob_lst = contract("ij,i->ij", prob_lst, prob_lst_norm).cpu().numpy()
    meas_lst = th.empty([m, dim, dim])
    if scheme == "g":
        #! vectorize better
        for i in range(m):
            measurement_res = np.random.choice(np.arange(0, dim), p=prob_lst[i])
            meas_lst[i] = th.tensor(
                sp.sparse.coo_matrix(
                    ([1], ([measurement_res], [measurement_res])), shape=(dim, dim)
                ).toarray(),
                dtype=th.cfloat,
            ).to(device)
        rho_hat_lst = (dim + 1) * contract(
            "oba,obc,ocd->oad", th.conj(U_lst), meas_lst, U_lst
        ) - th.eye(dim, dtype=th.cfloat).to(device)
    else:
        rho_hat_lst = th.empty([m, dim, dim], dtype=th.cfloat)
        #!vectorize
        for i in range(m):
            rho_hat = th.tensor([1], dtype=th.cfloat)
            measurement_res = list(
                np.binary_repr(
                    np.random.choice(np.arange(0, dim), p=prob_lst[i]), width=qubit_num
                )
            )
            measurement_res.reverse()
            i0 = th.tensor([[1, 0], [0, 0]], dtype=th.cfloat).to(device)
            i1 = th.tensor([[0, 0], [0, 1]], dtype=th.cfloat).to(device)
            for j in range(qubit_num):
                if measurement_res[j] == "0":
                    rho_hat_i = 3 * th.mm(
                        th.conj(U_lst_local[i][j]).T, th.mm(i0, U_lst_local[i][j]),
                    ) - th.eye(2).to(device)
                else:
                    rho_hat_i = 3 * th.mm(
                        th.conj(U_lst_local[i][j]).T, th.mm(i1, U_lst_local[i][j]),
                    ) - th.eye(2).to(device)
                rho_hat = th.kron(rho_hat, rho_hat_i)
            rho_hat_lst[i] = rho_hat
    return rho_hat_lst


def gen_u(n, qU):
    while True:
        qU.put(th.tensor(random_clifford(n).to_matrix(), dtype=th.cfloat))

randomU1=th.load("random1")
randomU2=th.load("random2")
def get_u(n,qU,num):
    if(n==2):
        return randomU2[np.random.randint(0, 11240,size=num)]
    if(n==1):
        return randomU1[np.random.randint(0, 24,size=num)]
    else:
        U_lst=th.empty(num,n**2,n**2)
        for i in range(num):
            U_lst[i]=qU[n].get()
    return U_lst
