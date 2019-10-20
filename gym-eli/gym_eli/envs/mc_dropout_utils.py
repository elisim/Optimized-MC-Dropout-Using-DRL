from tqdm import tqdm
import numpy as np


def mc_dropout(net, X_train, batch_size=1000, dropout=0.5, T=100):
    """
    net: keras model with set_mc_dropout_rate function

    Forward passes T times, then take the variance from all the predictions for each class.
    the mc_dropout score for an example will be the mean of the variances for all the classes.
    y_mc_dropout is the mean of all runs.
    """
    net.set_mc_dropout_rate(dropout)
    model = net.model
    repetitions = []
    # Todo: parallelize
    for _ in tqdm(range(T)):
        pred = model.predict(X_train, batch_size)
        repetitions.append(pred)

    net.set_mc_dropout_rate(0)
    preds = np.array(repetitions)  # T x data x pred

    # average over all passes
    y_mc_dropout = preds.mean(axis=0)

    # get variance from all preds for each example (output: batch x preds classes) each cell is var
    mc = np.var(preds, axis=0)
    # mean of vars of each class (out: one dim array with batch as dim)
    mc_uncertainty = np.mean(mc, axis=-1)

    return y_mc_dropout, -mc_uncertainty
