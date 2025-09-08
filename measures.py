from sklearn import metrics
import numpy as np


def eer(labels, sims):
    """Calculates Equal Error Rate (EER).

    :param labels: ground truths of examples (1 for same-speaker, 0 for different-speaker)
    :param sims: similarity scores of examples (higher is more similar)
    :return: EER and the score threshold at which EER is obtained
    """
    fprs, tprs, thresholds = metrics.roc_curve(labels, sims)
    fnrs = 1 - tprs
    closest_idx = np.nanargmin(np.absolute(fnrs - fprs))
    threshold = thresholds[closest_idx]
    fpr = fprs[closest_idx]
    fnr = fnrs[closest_idx]
    eer = (fpr + fnr) / 2

    return eer, threshold
