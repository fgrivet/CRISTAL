import numpy as np
from sklearn.metrics import auc


# Toolbox : Comparison metrics


def average_precision_score(y_true, y_score):
    score_outliers = y_score[y_true == -1]
    if len(score_outliers) == 0:
        return np.nan
    precision_scores = [precision(y_score, score_outliers, threshold) for threshold in score_outliers]
    return np.mean(precision_scores)


def roc_auc_score(y_true, y_score):
    if (res := roc_curve(y_true, y_score)) is None:
        return np.nan
    tpr, fpr, _ = res
    return auc(x=fpr, y=tpr)


def roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    score_outliers = y_score[y_true == -1]
    if len(score_outliers) == 0:
        return None
    score_inliers = y_score[y_true != -1]
    thresholds = np.unique(np.concatenate((np.array([np.min(y_score)]), score_outliers, np.array([np.max(y_score)]))))
    tpr = np.array([len(score_outliers[score_outliers < s]) / len(score_outliers) for s in thresholds])
    fpr = np.array([len(score_inliers[score_inliers < s]) / len(score_inliers) for s in thresholds])
    return tpr, fpr, thresholds


def precision(y_score, score_outliers, threshold):
    # len(y_score[y_score < threshold]) = 0
    # => there is no positives detected, which mean that the precision (how many positives are true positives) is max
    return len(score_outliers[score_outliers < threshold]) / len(y_score[y_score < threshold]) if len(y_score[y_score < threshold]) != 0 else 1


def roc_and_pr_curves(y_true, y_score):
    score_outliers = y_score[y_true == -1]
    score_inliers = y_score[y_true != -1]
    thresholds = np.concatenate((np.array([np.min(y_score)]), np.unique(score_outliers), [np.max(y_score)]))
    tpr = np.array([len(score_outliers[score_outliers < s]) / len(score_outliers) for s in thresholds])
    fpr = np.array([len(score_inliers[score_inliers < s]) / len(score_inliers) for s in thresholds])
    prec = np.array(
        [len(score_outliers[score_outliers < s]) / len(y_score[y_score < s]) if len(y_score[y_score < s]) != 0 else np.nan for s in thresholds]
    )
    return tpr, fpr, prec, thresholds


def supervised_metrics(y_true, y_pred):
    pred_outliers = y_pred[y_true == -1]
    pred_inliers = y_pred[y_true == 1]
    recall = len(pred_outliers[pred_outliers == -1]) / len(pred_outliers)  # TPR
    specificity = len(pred_inliers[pred_inliers == 1]) / len(pred_inliers)  # 1 - FPR
    precision_score = len(pred_outliers[pred_outliers == -1]) / len(y_pred[y_pred == -1]) if len(y_pred[y_pred == -1]) != 0 else np.nan
    accuracy = (recall + specificity) / 2
    f_score = 2 * recall * precision_score / (recall + precision_score) if recall + precision_score != 0 else 0
    return recall, specificity, precision_score, accuracy, f_score


def em_auc_score(scoring_func, samples, random_generator, n_generated: int = 100000):
    t_max = 0.9
    lim_inf = samples.min(axis=0)
    lim_sup = samples.max(axis=0)
    volume_support = (lim_sup - lim_inf).prod()
    t = np.linspace(0, 2 / (10 * volume_support), n_generated)
    unif = random_generator.uniform(lim_inf, lim_sup, size=(1000, 2))
    s_X = scoring_func(samples)
    s_unif = scoring_func(unif)
    res = em_goix(t, t_max, volume_support, s_unif, s_X, n_generated)
    return res[2]


def em_goix(t, t_max, volume_support: float, s_unif: np.ndarray, s_X: np.ndarray, n_generated: int):
    # copied from https://github.com/ngoix/EMMV_benchmarks
    EM_t = np.zeros(t.shape[0])
    n_samples = s_X.shape[0]
    s_X_unique = np.unique(s_X)
    EM_t[0] = 1.0
    for u in s_X_unique:
        # if (s_unif >= u).sum() > n_generated / 1000:
        EM_t = np.maximum(EM_t, 1.0 / n_samples * (s_X > u).sum() - t * (s_unif > u).sum() / n_generated * volume_support)
    if (amax := np.argmax(EM_t <= t_max) + 1) == 1:
        print("Failed to achieve t_max, values all greater than 0.9")
        amax = -1
    return t, EM_t, auc(x=t[:amax], y=EM_t[:amax]), amax


def mv_auc_score(scoring_func, samples, random_generator, n_generated: int = 100000):
    alpha_min = 0.9
    alpha_max = 0.999
    axis_alpha = np.linspace(alpha_min, alpha_max, n_generated)
    lim_inf = samples.min(axis=0)
    lim_sup = samples.max(axis=0)
    volume_support = (lim_sup - lim_inf).prod()
    unif = random_generator.uniform(lim_inf, lim_sup, size=(1000, 2))
    s_X = scoring_func(samples)
    s_unif = scoring_func(unif)
    res = mv_goix(axis_alpha, volume_support, s_unif, s_X, n_generated)
    return res[2]


def mv_goix(axis_alpha, volume_support: float, s_unif: np.ndarray, s_X: np.ndarray, n_generated: int):
    # copied from https://github.com/ngoix/EMMV_benchmarks
    n_samples = s_X.shape[0]
    s_X_argsort = s_X.argsort()
    mass = 0
    cpt = 0
    u = s_X[s_X_argsort[-1]]
    mv = np.zeros(axis_alpha.shape[0])
    for i in range(axis_alpha.shape[0]):
        while mass < axis_alpha[i]:
            cpt += 1
            u = s_X[s_X_argsort[-cpt]]
            mass = 1.0 / n_samples * cpt  # sum(s_X > u)
        mv[i] = float((s_unif >= u).sum()) / n_generated * volume_support
    return axis_alpha, mv, auc(x=axis_alpha, y=mv)
