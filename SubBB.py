from MasterSub import *


class SubBB:
    def __init__(self, sub, y_d, v_d, x_k, D):
        temp_sub, T, W, r, ub, lb, xx = sub.ReturnModel()
        T2 = {0: SubProb(temp_sub, T, W, r, ub, lb, xx)}  # Dictionary of active nodes in the seconds stage
        T2_v, T2_y = {0: v_d}, {0: y_d}
        TI2_v = {0: v_d}  # Dictionary saving nodes that are candidates for splitting
        D = D
        tt_last, d = 0, 1

        while (len(T2) < D, len(T2_v) != 0) == (True, True):
            tt_bar = min(T2_v, key=T2_v.get)
            ytc = {y_key: T2_y[tt_bar][y_key] for y_key in Y_int_keys}
            ynin = YInt(ytc)
            if len(ynin) == 0:
                del T2_v[tt_bar]
            else:
                ysk = SelectSplitKey(ytc, ynin)
                node_2, T_2, W_2, r_2, ub_2, lb_2, xx_2 = T2[tt_bar].ReturnModel()

                tt_last += 1
                T2[tt_last] = SubProb(node_2.copy(), copy.copy(T_2), copy.copy(W_2), copy.copy(r_2), copy.copy(ub_2), copy.copy(lb_2), copy.copy(xx_2))

                T2[tt_last].AddSplit(ysk, 'u')
                y2, v2 = T2[tt_last].FixXSolve(x_k)
                if y2 is not None:
                    T2_y[tt_last], T2_v[tt_last] = y2, v2
                    TI2_v[tt_last] = T2_v[tt_last]
                else:
                    del T2[tt_last]

                tt_last += 1
                T2[tt_last] = SubProb(node_2.copy(), copy.copy(T_2), copy.copy(W_2), copy.copy(r_2), copy.copy(ub_2), copy.copy(lb_2), copy.copy(xx_2))
                T2[tt_last].AddSplit(ysk, 'l')
                y2, v2 = T2[tt_last].FixXSolve(x_k)
                if y2 is not None:
                    T2_y[tt_last], T2_v[tt_last] = y2, v2
                    TI2_v[tt_last] = T2_v[tt_last]
                else:
                    del T2[tt_last]

                del T2[tt_bar]
                del T2_y[tt_bar]
                del T2_v[tt_bar]
                del TI2_v[tt_bar]

        self.T2 = T2
        self.T2_v = T2_v
