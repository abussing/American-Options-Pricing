import pandas as pd
import numpy as np
import time
import BroadieGlasserman
import CoxRossRubinstein
import LongstaffSchwartz


if __name__ == "__main__":

    # # table 1 from https://doi.org/10.1007/s10479-016-2267-4
    # isput = False  # call option
    # K = 80  # strike price
    # r = 0.06  # interest rate
    # T = 0.25  # time span
    # sigma = 0.4  # volatility parameter
    # q = 0.1  # continuous dividend yield
    # Svals = range(60, 105, 5)  # range of stock vals for our table
    # FDprices = [
    #     0.4166,
    #     1.0092,
    #     2.0647,
    #     3.6986,
    #     5.9366,
    #     8.8578,
    #     12.3121,
    #     16.2426,
    #     20.5556,
    # ]

    # # table 2 from https://doi.org/10.1007/s10479-016-2267-4
    # isput = True  # put option
    # K = 80  # strike price
    # r = 0.08  # interest rate
    # T = 0.25  # time span
    # sigma = 0.4  # volatility parameter
    # q = 0  # continuous dividend yield
    # Svals = range(60, 105, 5)  # range of stock vals for our table
    # FDprices = [
    #     20.0039,
    #     15.3839,
    #     11.4326,
    #     8.2030,
    #     5.6866,
    #     3.8155,
    #     2.4839,
    #     1.5732,
    #     0.9721,
    # ]

    # pricemat = pd.DataFrame(
    #     0,
    #     index=pd.Index(Svals, name="Stock Price"),
    #     columns=["BG (lower)", "BG (upper)", "BG", "CRR", "LS", "FD"],
    # )
    # pricemat.loc[:, "FD"] = FDprices

    # timemat = pd.DataFrame(
    #     0, index=pd.Index(Svals, name="Stock Price"), columns=["CRR", "BG", "LS"],
    # )

    # for idx in pricemat.index:
    #     print(idx)
    #     # Cox Ross Rubinstein
    #     start_time = time.time()
    #     pricemat.loc[idx, "CRR"] = CoxRossRubinstein.binomial_pricer(
    #         isPut=isput, S=idx, K=K, r=r, sigma=sigma, timesteps=100, T=T, q=q
    #     )
    #     timemat.loc[idx, "CRR"] = time.time() - start_time

    #     # Broadie Glasserman
    #     start_time = time.time()
    #     tempboundz = BroadieGlasserman.BGMC(
    #         b=20,  # how many branches
    #         n=25,  # how many times to repeat simulation
    #         stockprice_t0=idx,
    #         r=r,
    #         delta=q,
    #         sigma=sigma,
    #         T=T,
    #         timesteps=3,
    #         K=K,
    #         isput=isput,
    #     )
    #     timemat.loc[idx, "BG"] = time.time() - start_time

    #     pricemat.loc[idx, "BG (lower)"] = tempboundz[0]
    #     pricemat.loc[idx, "BG (upper)"] = tempboundz[1]
    #     pricemat.loc[idx, "BG"] = np.mean(tempboundz)

    #     # Schwarz Longstaff
    #     start_time = time.time()
    #     pricemat.loc[idx, "LS"] = LongstaffSchwartz.LSMC(
    #         b=10000,  # how many simulated paths
    #         stockprice_t0=idx,
    #         r=r,
    #         delta=q,
    #         sigma=sigma,
    #         T=T,
    #         timesteps=50,
    #         K=K,
    #         isput=isput,
    #     )
    #     timemat.loc[idx, "LS"] = time.time() - start_time

    # table 1 from https://doi.org/10.1093/rfs/14.1.113
    isput = True  # put option
    K = 40  # strike price
    r = 0.06  # interest rate
    q = 0  # continuous dividend yield
    FDprices = [
        4.478,
        4.840,
        7.101,
        8.508,
        3.250,
        3.745,
        6.148,
        7.67,
        2.314,
        2.885,
        5.312,
        6.92,
        1.617,
        2.212,
        4.582,
        6.248,
        1.11,
        1.69,
        3.948,
        5.647,
    ]
    S_use = [36] * 4 + [38] * 4 + [40] * 4 + [42] * 4 + [44] * 4
    sig_use = [0.2, 0.2, 0.4, 0.4] * 5
    T_use = [1, 2] * 10

    pricemat = pd.DataFrame(
        {
            "S": S_use,
            "sigma": sig_use,
            "T": T_use,
            "BG (lower)": [0] * 20,
            "BG (upper)": [0] * 20,
            "BG": [0] * 20,
            "CRR": [0] * 20,
            "LS": [0] * 20,
            "FD": [0] * 20,
        }
    )

    timemat = pd.DataFrame(
        {
            "S": S_use,
            "sigma": sig_use,
            "T": T_use,
            "BG": [0] * 20,
            "CRR": [0] * 20,
            "LS": [0] * 20,
        }
    )

    pricemat.loc[:, "FD"] = FDprices

    for j in range(0, pricemat.shape[0], 1):
        print(j)
        # Cox Ross Rubinstein
        start_time = time.time()
        pricemat.loc[j, "CRR"] = CoxRossRubinstein.binomial_pricer(
            isPut=isput,
            S=S_use[j],
            K=K,
            r=r,
            sigma=sig_use[j],
            timesteps=100,
            T=T_use[j],
            q=q,
        )
        timemat.loc[j, "CRR"] = time.time() - start_time

        # Broadie Glasserman
        start_time = time.time()
        tempboundz = BroadieGlasserman.BGMC(
            b=20,  # how many branches
            n=25,  # how many times to repeat simulation
            stockprice_t0=S_use[j],
            r=r,
            delta=q,
            sigma=sig_use[j],
            T=T_use[j],
            timesteps=3,
            K=K,
            isput=isput,
        )
        timemat.loc[j, "BG"] = time.time() - start_time

        pricemat.loc[j, "BG (lower)"] = tempboundz[0]
        pricemat.loc[j, "BG (upper)"] = tempboundz[1]
        pricemat.loc[j, "BG"] = np.mean(tempboundz)

        # longstaff schwartz
        start_time = time.time()
        pricemat.loc[j, "LS"] = LongstaffSchwartz.LSMC(
            b=10000,
            stockprice_t0=S_use[j],
            r=r,
            delta=q,
            sigma=sig_use[j],
            T=T_use[j],
            timesteps=50,
            K=K,
            isput=True,
        )
        timemat.loc[j, "LS"] = time.time() - start_time


print(pricemat)

print(timemat)


print("what")

