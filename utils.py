import numpy as np
import sklearn.metrics as skm
from sklearn.covariance import ledoit_wolf
from sklearn.cluster import KMeans
import tensorflow as tf


def get_roc_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    auroc = skm.roc_auc_score(labels, data)
    return auroc


def get_pr_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    aupr = skm.average_precision_score(labels, data)
    return aupr


def get_fpr(xin, xood):
    return np.sum(xood < np.percentile(xin, 95)) / len(xood)


def get_scores_one_cluster(ftrain, ftest, ft_ood, fv_ood, shrunkcov=False):
    if shrunkcov:
        print("Using ledoit-wolf covariance estimator.")
        def cov(x): return ledoit_wolf(x)[0]
    else:
        def cov(x): return np.cov(x.T, bias=True)

    dtest = np.sum(
        (ftest - np.mean(ftrain, axis=0, keepdims=True))
        * (
            np.linalg.pinv(cov(ftrain)).dot(
                (ftest - np.mean(ftrain, axis=0, keepdims=True)).T
            )
        ).T,
        axis=-1,
    )

    dt_ood = np.sum(
        (ft_ood - np.mean(ftrain, axis=0, keepdims=True))
        * (
            np.linalg.pinv(cov(ftrain)).dot(
                (ft_ood - np.mean(ftrain, axis=0, keepdims=True)).T
            )
        ).T,
        axis=-1,
    )

    dv_ood = np.sum(
        (fv_ood - np.mean(ftrain, axis=0, keepdims=True))
        * (
            np.linalg.pinv(cov(ftrain)).dot(
                (fv_ood - np.mean(ftrain, axis=0, keepdims=True)).T
            )
        ).T,
        axis=-1,
    )

    return dtest, dt_ood, dv_ood


def get_scores_save_features(ftrain, ftest, ft_ood, fv_ood, labelstrain, args_training_mode, args_clusters):
    if args_clusters == 1:
        return get_scores_one_cluster(ftrain, ftest, ft_ood, fv_ood)
    else:
        if args_training_mode == "SupCE":
            print("Using data labels as cluster since model is cross-entropy")
            ypred = labelstrain
        else:
            ypred = get_clusters(ftrain, args_clusters)
        return get_scores_multi_cluster_save_features(ftrain, ftest, ft_ood, fv_ood, ypred)


def get_clusters(ftrain, nclusters):
    kmeans = KMeans(init="random", n_clusters=nclusters,
                    n_init=10, max_iter=300, random_state=42)
    kmeans.fit(ftrain)
    cluster_labels = kmeans.labels_
    return cluster_labels


def get_clusters_ved(ftrain, nclusters):
    kmeans = KMeans(init="random", n_clusters=nclusters,
                    n_init=10, max_iter=300, random_state=42)
    kmeans.fit(ftrain)
    cluster_labels = kmeans.labels
    return cluster_labels


def get_scores_multi_cluster(ftrain, ftest, ft_ood, fv_ood, ypred):
    xc = [ftrain[ypred == i] for i in np.unique(ypred)]

    din = [
        np.sum(
            (ftest - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (ftest - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]

    dt_ood = [
        np.sum(
            (ft_ood - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (ft_ood - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]

    dv_ood = [
        np.sum(
            (fv_ood - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (fv_ood - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]

    din = np.min(din, axis=0)
    dt_ood = np.min(dt_ood, axis=0)
    dv_ood = np.min(dv_ood, axis=0)

    return din, dt_ood, dv_ood


def get_scores_multi_cluster_save_features(ftrain, ftest, ft_ood, fv_ood, ypred):
    xc = [ftrain[ypred == i] for i in np.unique(ypred)]

    din = [
        np.sum(
            (ftest - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (ftest - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]

    dt_ood = [
        np.sum(
            (ft_ood - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (ft_ood - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]

    dv_ood = [
        np.sum(
            (fv_ood - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (fv_ood - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]

    din = np.min(din, axis=0)
    dt_ood = np.min(dt_ood, axis=0)
    dv_ood = np.min(dv_ood, axis=0)

    return din, dt_ood, dv_ood, ypred


def get_eval_standardize_results_save_features(ftrain, ftest, ft_ood, fv_ood, labelstrain, args_training_mode, args_clusters):
    """
    standardize data and get evaluation results
    """
    ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
    ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
    ft_ood /= np.linalg.norm(ft_ood, axis=-1, keepdims=True) + 1e-10
    fv_ood /= np.linalg.norm(fv_ood, axis=-1, keepdims=True) + 1e-10

    m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(
        ftrain, axis=0, keepdims=True)

    ftrain = (ftrain - m) / (s + 1e-10)
    ftest = (ftest - m) / (s + 1e-10)
    ft_ood = (ft_ood - m) / (s + 1e-10)
    fv_ood = (fv_ood - m) / (s + 1e-10)

    if args_clusters == 1:
        dtest, dt_ood, dv_ood = get_scores_save_features(
            ftrain, ftest, ft_ood, fv_ood, labelstrain, args_training_mode, args_clusters)
    else:
        dtest, dt_ood, dv_ood, y_clusters = get_scores_save_features(
            ftrain, ftest, ft_ood, fv_ood, labelstrain, args_training_mode, args_clusters)

    dt_fpr95 = get_fpr(dtest, dt_ood)
    dt_auroc, dt_aupr = get_roc_sklearn(
        dtest, dt_ood), get_pr_sklearn(dtest, dt_ood)

    dv_fpr95 = get_fpr(dtest, dv_ood)
    dv_auroc, dv_aupr = get_roc_sklearn(
        dtest, dv_ood), get_pr_sklearn(dtest, dv_ood)

    if args_clusters == 1:
        return dt_fpr95, dt_auroc, dt_aupr, dv_fpr95, dv_auroc, dv_aupr, ftrain, ftest, ft_ood, fv_ood
    else:
        return dt_fpr95, dt_auroc, dt_aupr, dv_fpr95, dv_auroc, dv_aupr, ftrain, ftest, ft_ood, fv_ood, y_clusters


def get_eval_results_save_features_submit(ftrain, ftest, ft_ood, fv_ood, labelstrain, args_training_mode, args_clusters):
    """
    choose the scale types for evaluation results
    """
    ftrain = np.float32(ftrain)
    ftest = np.float32(ftest)
    ft_ood = np.float32(ft_ood)
    fv_ood = np.float32(fv_ood)

    if args_clusters == 1:
        _, _, _, dv_fpr95, dv_auroc, dv_aupr, ftrain, ftest, ft_ood, fv_ood = get_eval_standardize_results_save_features(
            ftrain, ftest, ft_ood, fv_ood, labelstrain, args_training_mode, args_clusters)
    else:
        _, _, _, dv_fpr95, dv_auroc, dv_aupr, ftrain, ftest, ft_ood, fv_ood, _ = get_eval_standardize_results_save_features(
            ftrain, ftest, ft_ood, fv_ood, labelstrain, args_training_mode, args_clusters)

    results = {
        'dv_fpr95': dv_fpr95,
        'dv_auroc': dv_auroc,
        'dv_aupr': dv_aupr
    }

    return results


def supervised_nt_xent_loss(z, y, temperature=0.5, base_temperature=0.07):
    '''
    Supervised normalized temperature-scaled cross entropy loss. 
    '''
    batch_size = tf.shape(z)[0]
    contrast_count = 1
    anchor_count = contrast_count
    y = tf.expand_dims(y, -1)

    mask = tf.cast(tf.equal(y, tf.transpose(y)), tf.float32)
    anchor_dot_contrast = tf.divide(
        tf.matmul(z, tf.transpose(z)),
        temperature
    )
    
    # for numerical stability
    logits_max = tf.reduce_max(anchor_dot_contrast, axis=1, keepdims=True)
    logits = anchor_dot_contrast - logits_max
    # # tile mask
    logits_mask = tf.ones_like(mask) - tf.eye(batch_size)
    mask = mask * logits_mask
    
    # compute log_prob
    exp_logits = tf.exp(logits) * logits_mask
    log_prob = logits - \
        tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True))

    mask_sum = tf.reduce_sum(mask, axis=1)
    mean_log_prob_pos = tf.reduce_sum(
        mask * log_prob, axis=1)[mask_sum > 0] / mask_sum[mask_sum > 0]

    loss = -(temperature / base_temperature) * mean_log_prob_pos
    loss = tf.reduce_mean(loss)
    return loss
