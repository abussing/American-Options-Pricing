import numpy as np
import pandas as pd
import copy


class StockPriceTree:
    def __init__(self):
        self.level = 0
        self.price = 1
        self.children = []
        self.parent = None

    @classmethod
    def exercisevalue(cls, current_stockprice, isput, K):
        return np.maximum(0, ((-1) ** isput) * (current_stockprice - K))

    @classmethod
    def discountedval(cls, undiscounted_val, r, T, timesteps):
        return np.exp(-r * (T / timesteps)) * undiscounted_val

    def create_children(self, num_children, depth, r, T, timesteps, sigma, delta):
        if depth == 0:
            return

        if self.parent is None:
            self.level = depth

        for i in range(num_children):
            child = StockPriceTree()
            child.parent = self
            child.level = depth - 1
            child.price = self.price * np.exp(
                (r - delta - ((sigma ** 2) / 2)) * (T / timesteps)
                + np.random.normal(loc=0, scale=sigma * np.sqrt(T / timesteps))
            )
            self.children.append(child)
            child.create_children(
                num_children, depth - 1, r, T, timesteps, sigma, delta
            )

    def eval_level(self, level, K, isput, is_upperbound, r, T, timesteps):
        if self.level == level:
            if level == 0:
                self.price = StockPriceTree.exercisevalue(self.price, isput, K)
            else:
                if is_upperbound:
                    childrenprices = [x.price for x in self.children]
                    exerciseval = StockPriceTree.exercisevalue(self.price, isput, K)
                    discounted_children = StockPriceTree.discountedval(
                        np.mean(childrenprices), r, T, timesteps,
                    )

                    self.price = np.maximum(exerciseval, discounted_children)
                else:
                    pricevec = np.zeros(len(self.children))
                    for leaveout in range(0, len(self.children), 1):
                        leftoutset = [x.price for x in self.children]
                        leftoutset.pop(leaveout)
                        discounthold_leftout = StockPriceTree.discountedval(
                            np.mean(leftoutset), r, T, timesteps,
                        )
                        exercisenow = StockPriceTree.exercisevalue(self.price, isput, K)
                        if discounthold_leftout > exercisenow:
                            pricevec[leaveout] = StockPriceTree.discountedval(
                                [x.price for x in self.children][leaveout],
                                r,
                                T,
                                timesteps,
                            )
                        else:
                            pricevec[leaveout] = exercisenow
                    self.price = np.mean(pricevec)
        for child in self.children:
            child.eval_level(level, K, isput, is_upperbound, r, T, timesteps)

    def eval_tree(self, K, isput, is_upperbound, r, T, timesteps):
        for lvl in range(0, self.level + 1, 1):
            self.eval_level(lvl, K, isput, is_upperbound, r, T, timesteps)

    def print_tree(self, timesteps):
        spaces = "   " * (timesteps - self.level)
        print(spaces + str(self.price) + "  " + str(self.level))
        for child in self.children:
            child.print_tree(timesteps)


def BGMC(b, n, stockprice_t0, r, delta, sigma, T, timesteps, K, isput):
    retvec = np.zeros(shape=(n, 2))
    for j in range(n):
        # CREATE THE TREE
        ourtree = StockPriceTree()
        ourtree.price = stockprice_t0
        ourtree.create_children(
            num_children=b,
            depth=timesteps,
            r=r,
            T=T,
            timesteps=timesteps,
            sigma=sigma,
            delta=delta,
        )

        # COPY THE TREE SO YOU CAN GET UPPER BOUND WITH 1 AND LOWER WITH OTHER
        ourtree_copy = copy.deepcopy(ourtree)

        # FIND LOWER AND UPPER BOUNDS
        ourtree.eval_tree(
            K=K, isput=isput, is_upperbound=False, r=r, T=T, timesteps=timesteps
        )
        ourtree_copy.eval_tree(
            K=K, isput=isput, is_upperbound=True, r=r, T=T, timesteps=timesteps
        )

        retvec[j, :] = np.array([ourtree.price, ourtree_copy.price])

    return np.mean(retvec, axis=0)


if __name__ == "__main__":
    n = 100
    b = 50
    T = 1
    timesteps = 3
    r = 0.05
    delta = 0.1
    sigma = 0.2
    K = 100

    outputmat = pd.DataFrame(
        {"Low Est": [0] * 7, "High Est": [0] * 7, "Point Estimate": [0] * 7},
        index=pd.Index(range(70, 140, 10), name="Stock Price"),
    )

    for stk in outputmat.index:

        tempboundz = BGMC(
            b=b,
            n=n,
            stockprice_t0=stk,
            r=r,
            delta=delta,
            sigma=sigma,
            T=T,
            timesteps=timesteps,
            K=K,
            isput=False,
        )

        outputmat.loc[stk, "Low Est"] = tempboundz[0]
        outputmat.loc[stk, "High Est"] = tempboundz[1]
        outputmat.loc[stk, "Point Estimate"] = np.mean(tempboundz)

    print(outputmat)

