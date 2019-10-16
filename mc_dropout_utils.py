import numpy as np
from tqdm import tqdm


def mc_dropout(net, X_test, batch_size=1000, dropout=0.5, T=100):
    """
    net: keras model with set_mc_dropout_rate function
    
    Forward passes T times, then take the variance from all the predictions for each class.
    the mc_dropout score for an example will be the mean of the variances for all the classes.  
    """
    net.set_mc_dropout_rate(dropout)
    model = net.model
    repititions = []
    # Todo: parallelize
    for _ in tqdm(range(T)):
        pred = model.predict(X_test, batch_size)
        repititions.append(pred)
    net.set_mc_dropout_rate(0)

    repititions = np.array(repititions) # T x btach x pred
    mc = np.var(repititions, axis=0) # get variance from all preds for each example (output: batch x preds classes) each cell is var
    mc = np.mean(mc, axis=-1) # mean of vars of each class (out: one dim array with batch as dim)
    return -mc


def evaluate_mc(net, X_test, y_test, mc_dropout_rate, sample_times=50):
    """
    net: keras model with set_mc_dropout_rate function
    
    create mc passes.
    avg the passes for each data.
    compare to real y and count the number of times of an error
    output mc error
    """
    net.set_mc_dropout_rate(mc_dropout_rate)
    model = net.model
    batch_size = 1000
    err = 0.
    for batch_id in tqdm(range(X_test.shape[0] // batch_size)):
        # take batch of data
        x = X_test[(batch_id*batch_size):((batch_id + 1)*batch_size)]
        # init empty predictions
        y_ = np.zeros((sample_times, batch_size, y_test[0].shape[0])) # mc preds: T x batch x preds (T:50 x Data:1000 x labels:10)

        for sample_id in range(sample_times):
            # save predictions from a sample pass
            y_[sample_id] = model.predict(x, batch_size) # for each pass, you have predictions for batch (1000x10)
        
        # average over all passes
        mean_y = y_.mean(axis=0) # get mean of preds for each data in the passes
        # evaluate against labels
        y = y_test[(batch_id*batch_size):((batch_id + 1)*batch_size)]
        # compute error
        err += np.count_nonzero(np.not_equal(mean_y.argmax(axis=1), y.argmax(axis=1))) # count the number of wrong classifications (label is 1 and i said 2 after mc)

    err = err / X_test.shape[0]
    net.set_mc_dropout_rate(0)

    return 1. - err