import pandas as pd
import numpy as np
import time
import BroadieGlasserman
import CoxRossRubinstein
import LongstaffSchwartz


if __name__ == "__main__":

    # table 1 from https://doi.org/10.1007/s10479-016-2267-4
    isput = False  # call option
    K = 80  # strike price
    r = 0.06  # interest rate
    T = 0.25  # time span
    sigma = 0.4  # volatility parameter
    q = 0.1  # continuous dividend yield
    Svals = range(60, 105, 5)  # range of stock vals for our table

    pricemat = pd.DataFrame(
        0,
        index=pd.Index(Svals, name="Stock Price"),
        columns=["CRR", "BG (lower)", "BG (upper)", "BG (point est.)", "LS"],
    )

    timemat = pd.DataFrame(
        0, index=pd.Index(Svals, name="Stock Price"), columns=["CRR", "BG", "LS"],
    )

    for idx in pricemat.index:
        print(idx)
        # Cox Ross Rubinstein
        start_time = time.time()
        pricemat.loc[idx, "CRR"] = CoxRossRubinstein.binomial_pricer(
            isPut=isput, S=idx, K=K, r=r, sigma=sigma, timesteps=100, T=T, q=q
        )
        timemat.loc[idx, "CRR"] = time.time() - start_time

        # Broadie Glasserman
        start_time = time.time()
        tempboundz = BroadieGlasserman.BGMC(
            b=10,  # how many branches
            n=10,  # how many times to repeat simulation
            stockprice_t0=idx,
            r=r,
            delta=q,
            sigma=sigma,
            T=T,
            timesteps=3,
            K=K,
            isput=isput,
        )
        timemat.loc[idx, "BG"] = time.time() - start_time

        pricemat.loc[idx, "BG (lower)"] = tempboundz[0]
        pricemat.loc[idx, "BG (upper)"] = tempboundz[1]
        pricemat.loc[idx, "BG (point est.)"] = np.mean(tempboundz)

        # Schwarz Longstaff
        start_time = time.time()
        pricemat.loc[idx, "LS"] = LongstaffSchwartz.LSMC(
            b=10000,  # how many simulated paths
            stockprice_t0=idx,
            r=r,
            delta=q,
            sigma=sigma,
            T=T,
            timesteps=50,
            K=K,
            isput=isput,
        )
        timemat.loc[idx, "LS"] = time.time() - start_time


print(pricemat)

print(timemat)


print("what")

