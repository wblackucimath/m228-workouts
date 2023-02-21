from seaborn import histplot
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm, bernoulli, binom
from importlib_resources import files
import numpy as np


def solution_4(N_sim=10**4, N_sum=10**4, show_plt=False, save_plt=True, fname=None):
    plt.rcParams["text.usetex"] = True
    plt.figure(figsize=(8, 4.5))

    if True:
        data = (
            2
            * binom.rvs(
                size=N_sim,
                n=N_sum,
                p=0.5,
            )
            - N_sum
        ) / np.sqrt(N_sum)
        # Binomial random variable is the sum of i.i.d. Bernoulli random variables
    else:
        data = np.sum(
            2 * bernoulli.rvs(p=0.5, size=(N_sim, N_sum)) - 1,
            axis=1,
        ) / np.sqrt(N_sum)

    def kde(x):
        return gaussian_kde(data).pdf(x.T)

    def pdf(x):
        return norm.pdf(
            x,
            loc=np.mean(data),
            scale=np.std(data),
        )

    X_plot = np.linspace(data.min(), data.max(), 500)

    plt.plot(
        X_plot,
        norm.pdf(X_plot),
        c="green",
    )
    plt.plot(
        X_plot,
        kde(X_plot),
        c="orange",
    )
    plt.plot(
        X_plot,
        pdf(X_plot),
        c="red",
    )
    histplot(
        data=data,
        stat="density",
    )

    plt.grid()

    plt.title(
        "The Distribution for "
        r"$\displaystyle{"
        + r"\frac{1}{\sqrt{"
        + f"n"
        + r"}}"
        + r"\sum_{n = 1}^{"
        + f"n"
        + r"}"
        + r"X_n"
        + r"}$"
        + r" with $n = "
        + f"{N_sum:.2e}"
        + r"$ "
        + r"and $"
        + f"{N_sim:.2e}"
        + r"$ samples."
    )

    plt.xlabel(
        "Value of "
        + r"$\displaystyle{"
        + r"\frac{1}{\sqrt{"
        + f"n"
        + r"}}"
        + r"\sum_{n = 1}^{"
        + f"n"
        + r"}"
        + r"X_n"
        + r"}$"
        + r" with $n = "
        + f"{N_sum:.2e}"
        + r"$"
    )
    plt.ylabel("Density")

    plt.legend(
        [
            f"Standard Normal Gaussian",
            f"Gaussian Kernel Density Estimate",
            f"Naive Gaussian from Data\n Mean: ${float(np.mean(data)):.2e}$\n $\\sigma^2$: ${float(np.std(data)):.2e}$",
            f"Value Density",
        ],
        loc="upper left",
    )

    plt.tight_layout()

    if save_plt:
        if fname is None:
            fname = "CLT.png"
        plt.savefig(files("m228_workouts.plots").joinpath(fname))
    if show_plt:
        plt.show()

    plt.close()
