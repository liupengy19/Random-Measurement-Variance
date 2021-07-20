import math
import matplotlib as mpl
from scipy.optimize import leastsq
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt


def draw1(qubit_number_lst, ls, gs, lr, gr):
    ls_log = [math.log(var, 2) for var in ls]
    gs_log = [math.log(var, 2) for var in gs]
    lr_log = [math.log(var, 2) for var in lr]
    gr_log = [math.log(var, 2) for var in gr]
    plt.figure()

    plt.scatter(qubit_number_lst, ls_log)
    plt.plot(qubit_number_lst, ls_log, label="Local Shadow")

    plt.scatter(qubit_number_lst, gs_log)
    plt.plot(qubit_number_lst, gs_log, label="Global Shadow")

    plt.scatter(qubit_number_lst, lr_log)
    plt.plot(qubit_number_lst, lr_log, label="Local Random")

    plt.scatter(qubit_number_lst, gr_log)
    plt.plot(qubit_number_lst, gr_log, label="Global Random")

    plt.xlabel("Qubit Number")
    plt.ylabel("Measurement Variance")

    plt.legend()
    plt.savefig("test.png")


def func(p, x):
    k, b = p
    return k * x + b


def error(p, x, y):
    return func(p, x) - y


def draw2(y, x):
    y_log = np.asarray([math.log(var, 2) for var in y])
    plt.figure()
    x_log = np.asarray([math.log(u, 2) for u in x])
    p0 = [1, 1]
    print(leastsq(error, p0, args=(x_log, y_log)))
    plt.scatter(x_log, y_log)
    plt.plot(x_log, y_log, label="Global Random")

    plt.xlabel("unitary Number")
    plt.ylabel("Measurement Variance")
    plt.legend()
    plt.savefig("test.png")
