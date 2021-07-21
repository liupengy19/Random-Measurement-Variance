from utils import *

# M is the measurement times, A and B contain same number of qubits
def shadow_m4_estimation(rho_hat_lst, na):
    r = th.stack([realignment(rho_hat, na) for rho_hat in rho_hat_lst])
    r_dg = th.conj(contract("ijk->ikj", r))
    m = len(rho_hat_lst)
    res = contract(
        "iab,jbc,kcd,lda->ijkl", r, r_dg, r, r_dg
    ).type(th.cdouble)  # use inclusion–exclusion principle to eliminate expressions like if(i==j)
    sum0 = contract("ijkl->", res)  # summation
    sum11 = (
        contract("aabb", res) + contract("abab", res) + contract("abba", res)
    )  # two pairs of indexes are the same
    sum1 = (
        contract("aabc->", res)
        + contract("abac->", res)
        + contract("abca->", res)
        + contract("baac->", res)
        + contract("baca->", res)
        + contract("cbaa->", res)
    )  # one pair
    sum2 = (
        contract("aaab->", res)
        + contract("aaba->", res)
        + contract("abaa->", res)
        + contract("baaa->", res)
    )  # a triple are the same
    sum3 = contract("aaaa", res)  # all four
    res_sum = sum0 - sum1 + 2 * sum2 - 6 * sum3 + sum11
    return float(res_sum.real / (m * (m - 1) * (m - 2) * (m - 3)))


def random_m4_estimation(rho, na, NuA, NuB, Nm, device, scheme, N_average, N_iid, qU):
    dim = len(rho)
    qubit_num = int(math.log(dim, 2))
    nb = qubit_num - na
    if scheme == "l":
        UAi_lst = (
            get_u(1, None, na * N_iid * N_average * NuA)
            .to(device)
            .view(na, N_iid, N_average, NuA, 2, 2)
        )
        UBi_lst = (
            get_u(1, None, nb * N_iid * N_average * NuA)
            .to(device)
            .view(nb, N_iid, N_average, NuB, 2, 2)
        )
        UA_lst = UAi_lst[0]
        for i in range(1, na):
            UA_lst = contract("abcde,abcgf->abcdgef", UAi_lst[i], UA_lst).view(
                N_iid, N_average, NuA, 2 ** (i + 1), 2 ** (i + 1)
            )
        UB_lst = UBi_lst[0]
        for i in range(1, nb):
            UB_lst = contract("abcde,abcgf->abcdgef", UBi_lst[i], UB_lst).view(
                N_iid, N_average, NuB, 2 ** (i + 1), 2 ** (i + 1)
            )
    else:
        UA_lst = (
            get_u(na, qU, N_iid * N_average * NuA)
            .view(N_iid, N_average, NuA, 2**na , 2**na )
            .to(device)
        )
        UB_lst = (
            get_u(nb, qU, N_iid * N_average * NuB)
            .view(N_iid, N_average, NuB, 2**nb , 2**nb)
            .to(device)
        )
    U = contract("orijk,orlmn->oriljmkn", UA_lst, UB_lst).view(
            [N_iid, N_average, NuA, NuB, dim, dim]
        )
    prob_ls = contract("orijab,bc,orijac->orija", U, rho, th.conj(U)).real
    if Nm != -1:
        meas_cdf = prob_ls.view(N_iid * N_average * NuA * NuB, dim).cumsum(1)
        randchoice=th.cuda.FloatTensor(Nm, N_iid * N_average * NuA * NuB, 1).uniform_()
        choices = [(randchoice[i] < meas_cdf).type(th.int8).argmax(axis=1)+dim*th.arange(N_iid * N_average * NuA * NuB).to(device) for i in range(Nm)]
        prob_lsi=th.sparse_coo_tensor(
                    choices[0].view(1,-1),
                    th.ones(N_iid * N_average * NuA * NuB).to(device),
                    (N_iid * N_average * NuA * NuB*dim,)
                )
        for i in range(1,Nm):
            prob_lsi+=th.sparse_coo_tensor(
                    choices[i].view(1,-1),
                    th.ones(N_iid * N_average * NuA * NuB).to(device) ,
                    (N_iid * N_average * NuA * NuB*dim,)
                )
        prob_ls=prob_lsi.to(device).to_dense().view(N_iid, N_average, NuA, NuB, dim).type(th.cfloat)/Nm
    prob_ls=prob_ls.type(th.cfloat)
    hm1 = hamming_distance_table(qubit_num, nb, na, scheme).to(device)
    hm2 = hamming_distance_table(qubit_num, 0, nb, scheme).to(device)
    sum0 = contract(
        "kl,mn,km,ln,orijk,oriql,orpjm,orpqn->o",
        hm1,
        hm1,
        hm2,
        hm2,
        prob_ls,
        prob_ls,
        prob_ls,
        prob_ls,
    )
    sum1 = contract(
        "kl,mn,km,ln,orijk,oriql,orijm,oriqn->o",
        hm1,
        hm1,
        hm2,
        hm2,
        prob_ls,
        prob_ls,
        prob_ls,
        prob_ls,
    )
    sum2 = contract(
        "kl,mn,km,ln,orijk,orijl,orpjm,orpjn->o",
        hm1,
        hm1,
        hm2,
        hm2,
        prob_ls,
        prob_ls,
        prob_ls,
        prob_ls,
    )

    sum3 = contract(
        "kl,mn,km,ln,orijk,orijl,orijm,orijn->o",
        hm1,
        hm1,
        hm2,
        hm2,
        prob_ls,
        prob_ls,
        prob_ls,
        prob_ls,
    )

    sum0 += sum3 - sum1 - sum2
    return (dim ** 2) * sum0.real / ((NuA * NuB * (NuA - 1) * (NuB - 1)) * N_average)


def shadow_m4_variance(rho, na, m, N_iid, device, scheme, qU):
    rho = rho.to(device)
    r = realignment(rho, na)
    real_value = th.trace(th.mm(th.mm(r, th.conj(r).T), th.mm(r, th.conj(r).T)))
    res = 0
    for _ in range(N_iid):
        predict = shadow_m4_estimation(shadow_estimator(rho, device, scheme, m, qU), na)
        res += (predict - real_value) ** 2
    return float(res.real / N_iid)


def random_m4_variance(rho, na, NuA, NuB, Nm, N_iid, device, scheme, qU,N_average=1):
    rho = rho.to(device)
    r = realignment(rho, na)
    real_value = th.trace(th.mm(th.mm(r, th.conj(r).T), th.mm(r, th.conj(r).T)))
    predict = random_m4_estimation(
        rho, na, NuA, NuB, Nm, device, scheme, N_average, N_iid, qU
    )
    res = ((predict - real_value) ** 2).real.sum() / N_iid
    return float(res)
