import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd


def exercisevalue(current_stockprice, isput, K):
    return np.maximum(0, ((-1) ** isput) * (current_stockprice - K))


def discountedval(undiscounted_val, r, T, timesteps):
    return np.exp(-r * (T / timesteps)) * undiscounted_val


def laguerre4(xvec):
    L_1 = xvec
    L_2 = xvec ** 2
    L_3 = xvec ** 3
    return np.c_[L_1, L_2, L_3]


def expectval_hold(holdpayoff, current_stockprice):
    y = holdpayoff
    X = laguerre4(current_stockprice)
    reg = LinearRegression().fit(X, y)

    return reg.predict(X)


def hold_or_exercise(current_stockprice, optionval_tplus1, isput, K, r, T, timesteps):
    ifexercisednow = exercisevalue(current_stockprice, isput, K)

    inthemoney = ifexercisednow > 0

    optionval_current = discountedval(optionval_tplus1, r, T, timesteps)

    if sum(inthemoney) == 0:
        lsq_preds = -999
    else:
        lsq_preds = expectval_hold(
            holdpayoff=discountedval(optionval_tplus1[inthemoney], r, T, timesteps),
            current_stockprice=current_stockprice[inthemoney],
        )

    optionval_current[inthemoney] = (
        (lsq_preds > ifexercisednow[inthemoney])
        * discountedval(
            undiscounted_val=optionval_tplus1[inthemoney], r=r, T=T, timesteps=timesteps
        )
    ) + (lsq_preds <= ifexercisednow[inthemoney]) * ifexercisednow[inthemoney]

    return optionval_current


def pathgenerator(b, stockprice_t0, r, delta, sigma, T, timesteps):
    steps_use = np.random.normal(
        loc=0, scale=sigma * np.sqrt(T / timesteps), size=(b, timesteps + 1)
    )

    steps_anti = -1 * steps_use
    normsteps = np.concatenate((steps_use, steps_anti), axis=0)
    normsteps = normsteps + (r - delta - ((sigma ** 2) / 2)) * (T / timesteps)

    normsteps[:, 0] = 0

    S_lognormpath = np.exp(np.cumsum(normsteps, axis=1))
    S_lognormpath[:, 0] = 1

    S_lognormpath = stockprice_t0 * S_lognormpath

    return S_lognormpath


def LSMC(b, stockprice_t0, r, delta, sigma, T, timesteps, K, isput):
    simpaths = pathgenerator(b, stockprice_t0, r, delta, sigma, T, timesteps)

    simpaths[:, timesteps] = exercisevalue(
        current_stockprice=simpaths[:, timesteps], K=K, isput=isput
    )
    for t_i in range(timesteps - 1, 0, -1):

        simpaths[:, t_i] = hold_or_exercise(
            current_stockprice=simpaths[:, t_i],
            optionval_tplus1=simpaths[:, t_i + 1],
            isput=isput,
            K=K,
            r=r,
            T=T,
            timesteps=timesteps,
        )

    return np.mean(
        discountedval(undiscounted_val=simpaths[:, 1], r=r, T=T, timesteps=timesteps)
    )


if __name__ == "__main__":

    S_use = [36] * 4 + [38] * 4 + [40] * 4 + [42] * 4 + [44] * 4
    sig_use = [0.2, 0.2, 0.4, 0.4] * 5
    T_use = [1, 2] * 10

    outputmat = pd.DataFrame(
        {"S": S_use, "sigma": sig_use, "T": T_use, "Simulated American": [0] * 20}
    )

    for j in range(0, outputmat.shape[0], 1):

        outputmat.loc[j, "Simulated American"] = LSMC(
            b=50000,
            stockprice_t0=S_use[j],
            r=0.06,
            delta=0,
            sigma=sig_use[j],
            T=T_use[j],
            timesteps=50,
            K=40,
            isput=True,
        )

    print(outputmat)

