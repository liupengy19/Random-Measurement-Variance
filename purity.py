from utils import *


def Shadow_Purity_Estimation(rho_hat_lst):  # M is the measurement times
    res = 0
    M = len(rho_hat_lst)
    for i in range(M):
        for j in range(i + 1, M):
            res += (rho_hat_lst[i] * rho_hat_lst[j].T).sum().real
    return res * 2 / (M * (M - 1))


# M stands for measurement times, and times is the calculation times
def Local_Shadow_Purity_Variance(rho, M, times, device):
    real_value = (rho * rho.T).sum().real.to(device)
    res = 0
    for i in range(times):
        res += (
            Shadow_Purity_Estimation(
                [Local_Shadow_Estimator(rho, device) for j in range(M)]
            )
            - real_value
        ) ** 2 / times
    return res


# M stands for measurement times, and times is the calculation times
def Global_Shadow_Purity_Variance(rho, M, times, device):
    real_value = (rho * rho.T).sum().real.to(device)
    res = 0
    for i in range(times):
        res += (
            Shadow_Purity_Estimation(
                [Global_Shadow_Estimator(rho, device) for j in range(M)]
            )
            - real_value
        ) ** 2 / times
    return res


def Local_Random_Purity_Estimation(rho, Nu, Nm, device):  # Set Nu = 1
    n = int(math.log(len(rho), 2))
    res = 0

    for i in range(Nu):
        measurement_res = []  # recording of measurement results
        U = torch.tensor(random_clifford(1).to_matrix(), dtype=torch.cfloat).to(device)
        for i in range(n - 1):
            u = torch.tensor(random_clifford(1).to_matrix(), dtype=torch.cfloat).to(
                device
            )
            U = torch.kron(U, u)
        evoluted_rho = torch.mm(U, torch.mm(rho, torch.conj(U).T))

        prob_lst = torch.diagonal(evoluted_rho).real
        prob_lst = prob_lst + torch.abs(prob_lst)
        prob_lst = prob_lst / torch.linalg.norm(prob_lst, 1)
        prob_lst = prob_lst.cpu().numpy()
        for j in range(Nm):
            measurement_res.append(np.random.choice(np.arange(0, 2 ** n), p=prob_lst))

        for j in range(Nm):
            j_str = Ddigit_Represent(measurement_res[j], 2, n)
            for k in range(j + 1, Nm):
                k_str = Ddigit_Represent(measurement_res[k], 2, n)
                res += 2 ** n * (-2) ** (-Hamming_Distance(j_str, k_str))
    return res * 2 / (Nm * (Nm - 1)) / Nu


def Global_Random_Purity_Estimation(rho, Nu, Nm, device):  # Set Nu = 1
    dim = len(rho)
    n = int(math.log(len(rho), 2))
    res = 0

    for i in range(Nu):
        measurement_res = []  # recording of measurement results
        U = torch.tensor(random_clifford(n).to_matrix(), dtype=torch.cfloat).to(device)
        evoluted_rho = torch.mm(U, torch.mm(rho, torch.conj(U).T))
        prob_lst = torch.diagonal(evoluted_rho).real
        prob_lst = prob_lst + torch.abs(prob_lst)
        prob_lst = prob_lst / torch.linalg.norm(prob_lst, 1)
        prob_lst = prob_lst.cpu().numpy()
        for j in range(Nm):
            measurement_res.append(np.random.choice(np.arange(0, 2 ** n), p=prob_lst))

        for j in range(Nm):
            for k in range(j + 1, Nm):
                if measurement_res[j] == measurement_res[k]:
                    res += dim
                else:
                    res -= 1
    return res * 2 / (Nm * (Nm - 1)) / Nu


def Local_Random_Purity_Variance(rho, Nu, Nm, times, device):
    res = 0
    real_value = (rho * rho.T).sum().real
    for i in range(times):
        res += (
            Local_Random_Purity_Estimation(rho, Nu, Nm, device) - real_value
        ) ** 2 / times
    return res


def Global_Random_Purity_Variance(rho, Nu, Nm, times, device):
    res = 0
    real_value = (rho * rho.T).sum().real
    for i in range(times):
        res += (
            Global_Random_Purity_Estimation(rho, Nu, Nm, device) - real_value
        ) ** 2 / times
    return res


def runPurityVar(qubit_number_lst, times, ls, gs, lr, gr, device):
    Local_Shadow_Purity_Variance_lst = []
    Global_Shadow_Purity_Variance_lst = []
    Local_Random_Purity_Variance_lst = []
    Global_Random_Purity_Variance_lst = []
    Nu = 5
    Nm = 10
    M = 5
    for n in qubit_number_lst:
        rho = noisy_W(n, 0)
        Local_Shadow_Purity_Variance_lst.append(
            Local_Shadow_Purity_Variance(rho, M, times, device).cpu().numpy()
        )
        Global_Shadow_Purity_Variance_lst.append(
            Global_Shadow_Purity_Variance(rho, M, times, device).cpu().numpy()
        )
        Local_Random_Purity_Variance_lst.append(
            Local_Random_Purity_Variance(rho, Nu, Nm, times, device).cpu().numpy()
        )
        Global_Random_Purity_Variance_lst.append(
            Global_Random_Purity_Variance(rho, Nu, Nm, times, device).cpu().numpy()
        )
    ls.put(Local_Shadow_Purity_Variance_lst)
    gs.put(Global_Shadow_Purity_Variance_lst)
    lr.put(Local_Random_Purity_Variance_lst)
    gr.put(Global_Random_Purity_Variance_lst)
