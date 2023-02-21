import yfinance as yf
from seaborn import displot, histplot
import matplotlib.pyplot as plt


def solution_3(period="2y"):
    price_df = yf.download("^GSPC", period=period, interval="1d")[["Close"]]
    R = price_df.pct_change()

    plot = histplot(
        R,
        stat="density",
        x="Close",
        kde=True,
        legend=True,
    )
    plt.xlabel("Return Amount")
    plt.ylabel("Density")
    plot.legend(
        [
            "Kernel Density Estimate",
            "Return Amount",
        ]
    )
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    solution_3("10y")
