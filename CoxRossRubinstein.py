import numpy as np
import pandas as pd


def build_stockpricetree(S_initial, d, u, N_periods):
    # MAKES A MATRIX WITH POSSIBLE STOCK PRICES ON LOWER TRIANGULAR PORTION
    mastervec = S_initial * (d ** np.arange(start=0, stop=N_periods + 1, step=1))
    stockpricemat = np.zeros(shape=(N_periods + 1, N_periods + 1))

    for col in range(0, stockpricemat.shape[0], 1):
        stockpricemat[col:, col] = mastervec[0 : len(mastervec) - col] * (u ** col)

    return stockpricemat


def binomial_pricer(isPut, S, K, r, sigma, timesteps, T, q=0, D=0, D_time=-1):
    delta_t = T / timesteps
    u = np.exp(sigma * np.sqrt(delta_t))
    d = np.exp(-sigma * np.sqrt(delta_t))
    p = (np.exp((r - q) * delta_t) - d) / (u - d)

    def optionvalue(Cd, Cu, S_here):
        C = np.exp(-r * delta_t) * (p * Cu + (1 - p) * Cd)
        return np.maximum(C, ((-1) ** isPut) * (S_here - K))

    # CREATE MATRIX OF POSSIBLE STOCK PRICES
    stockpricemat = build_stockpricetree(
        S_initial=S - np.exp(-r * D_time) * D, d=d, u=u, N_periods=timesteps
    )

    # CALCULATE FINAL VALUE OF OPTION BASED ON POSSIBLE STOCK PRICES AT FINAL TIMESTEP
    optionpricemat = np.zeros(shape=(timesteps + 1, timesteps + 1))
    optionpricemat[-1, :] = optionvalue(Cd=0, Cu=0, S_here=stockpricemat[-1, :],)

    # WORKING BACKWARDS, FILL IN PRICES FOR REST OF TIMESTEPS
    for rw in range(optionpricemat.shape[0] - 2, -1, -1):
        for col in range(0, rw + 1, 1):
            if rw * delta_t < D_time:
                optionpricemat[rw, col] = optionvalue(
                    Cd=optionpricemat[rw + 1, col],
                    Cu=optionpricemat[rw + 1, col + 1],
                    S_here=stockpricemat[rw, col]
                    + D * np.exp(-r * (D_time - rw * delta_t)),
                )
            else:
                optionpricemat[rw, col] = optionvalue(
                    Cd=optionpricemat[rw + 1, col],
                    Cu=optionpricemat[rw + 1, col + 1],
                    S_here=stockpricemat[rw, col],
                )

    return optionpricemat[0, 0]


if __name__ == "__main__":

    outputmat = pd.DataFrame(
        {"Binomial": [0] * 9}, index=pd.Index(range(60, 105, 5), name="Stock Price")
    )

    for idx in outputmat.index:
        outputmat.loc[idx, "Binomial"] = binomial_pricer(
            isPut=False, S=idx, K=80, r=0.06, sigma=0.4, timesteps=1000, T=0.25, q=0.1
        )

    print(outputmat)
