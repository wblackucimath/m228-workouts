import yfinance as yf
from seaborn import histplot
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from importlib_resources import files
from numpy import linspace


def solution_3(period="2y", show_plt=False, save_plt=True, fname=None):
    plt.rcParams["text.usetex"] = True
    plt.figure(figsize=(8, 4.5))
    price_df = yf.download(
        "^GSPC",
        period=period,
        interval="1d",
    )[["Close"]]

    R = price_df.pct_change().dropna()

    def kde(x):
        return gaussian_kde(R.values.T).pdf(x.T)

    def pdf(x):
        return norm.pdf(
            x,
            loc=R.mean(),
            scale=R.std(),
        )

    X_plot = linspace(R.min(), R.max(), 500)

    plt.plot(
        X_plot,
        kde(X_plot),
        c="orange",
    )
    plt.plot(
        X_plot,
        pdf(X_plot),
        c="r",
    )
    histplot(
        data=R,
        stat="density",
        x="Close",
    )

    plt.grid()

    plt.title("SPX Data with Gaussian Kernel Density Estimator")

    plt.xlabel("Return Amount")
    plt.ylabel("Density")

    plt.legend(
        [
            f"Gaussian Kernel Density Estimate",
            f"Naive Gaussian from Data\n (Mean: ${float(R.mean()):.2e}$, $\\sigma^2 = {float(R.std()):.2e}$)",
            f"Return Amount",
        ],
        loc="upper left",
    )

    plt.tight_layout()

    if save_plt:
        if fname is None:
            fname = "SPX_Histogram.png"
        plt.savefig(files("m228_workouts.plots").joinpath(fname))
    if show_plt:
        plt.show()

    plt.close()

    
    return float(R.mean()), float(R.std())
