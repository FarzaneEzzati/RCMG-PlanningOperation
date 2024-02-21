from MasterSub import *


def SubBB(sub, y_d, v_d, x_k, D):
    temp_sub, xx, yy, yitk = sub.ReturnModel()
    T2 = {0: SubProb(temp_sub, xx, yy, yitk)}  # Dictionary of active nodes in the seconds stage
    T2_v, T2_y = {0: v_d}, {0: y_d}
    TI2_v = {0: v_d}  # Dictionary saving nodes that are candidates for splitting
    tt_last, d = 0, 1

    while (len(T2) < D, len(TI2_v) != 0) == (True, True):
        tt_bar = min(TI2_v, key=TI2_v.get)
        ytc = {y_key: T2_y[tt_bar][y_key] for y_key in yitk}
        ynin = YInt(ytc)
        if len(ynin) == 0:
            del TI2_v[tt_bar]
        else:
            ysk = SelectSplitKey(ytc, ynin)
            node_2, xx_2, yy_2, yitk_2 = T2[tt_bar].ReturnModel()

            tt_last += 1
            T2[tt_last] = SubProb(node_2.copy(), copy.copy(xx_2), copy.copy(yy_2), copy.copy(yitk_2))

            T2[tt_last].AddSplit(ysk, 'u')
            y2, v2 = T2[tt_last].FixXSolve(x_k)
            if y2 is not None:
                T2_y[tt_last], T2_v[tt_last] = y2, v2
                TI2_v[tt_last] = T2_v[tt_last]
            else:
                del T2[tt_last]

            tt_last += 1
            T2[tt_last] = SubProb(node_2.copy(), copy.copy(xx_2), copy.copy(yy_2), copy.copy(yitk_2))
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

    return T2

