import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from src.pot import calc_point2point
from tqdm import tqdm


# not used
def thresholding_algorithm(test, reconstructed, labels):
    # shapes of the input tensors
    n = 500  # scalar
    accuracy = np.zeros(n)  # shape: (N,)
    threshold = 0.0  # scalar
    st = 0.0  # scalar
    pre = 1.0  # scalar
    m = 5  # assuming 5 iterations needed to reach the precision level

    for _ in range(m):
        for i in range(n):
            accuracy[i] = 0
            threshold = st + (i / pre)
            mae = np.mean(np.abs(test, reconstructed), axis=(1, 2))
            result = (mae < threshold) == np.where(labels[:, 0])
            accuracy[i] = np.sum(result)
            threshold = st + (np.argmax(accuracy) / pre)
        en = st + ((np.argmax(accuracy) + 1) / pre)
        st = st + ((np.argmax(accuracy) - 1) / pre)
        pre = pre * 10
        n = int((en - st) * pre)

    return threshold


def find_optimal_threshold(mean_score, labels):
    # Create an array of potential threshold values
    thresholds = np.arange(0.0001, 0.004, 0.00001)

    best_f1 = 0
    best_threshold = 0

    for threshold in tqdm(thresholds):
        preds = mean_score > threshold
        actual = labels > 0

        current_f1 = f1_score(actual, preds)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return best_threshold, best_f1


def eval(score, label, multi=False):
    if multi:
        label_main_attack = label[:, 1]
        label = label[:, 0]

    mean_score = np.mean(score, axis=(0, 2))
    roc_score = roc_auc_score(label > 0, mean_score)
    threshold, f1 = find_optimal_threshold(mean_score, label)
    print(f'Best f1: {f1}, Thresh: {threshold}, ROC: {roc_score}')

    preds = mean_score > threshold

    p_t = calc_point2point(preds, np.where(label > 0, 1, 0))

    metrics = {
        'f1': p_t[0],
        'precision': p_t[1],
        'recall': p_t[2],
        'TP': p_t[3],
        'TN': p_t[4],
        'FP': p_t[5],
        'FN': p_t[6],
        'ROC/AUC': p_t[7],
        'accuracy': p_t[8],
        'threshold': threshold,
        # 'pot-latency': p_latency
    }

    if multi:
        label_types = np.unique(label_main_attack)
        for lb in label_types:
            if lb == 0:
                mask = np.ones_like(label, dtype=bool)
            else:
                mask = label_main_attack == lb
            pred_only_curr_lb = preds[mask]
            if lb == 0:
                cond = label[mask] == 0
                pred_only_curr_lb = 1 - pred_only_curr_lb
            else:
                cond = label[mask] > 0

            actual = np.where(cond, 1, 0)
            p_t = calc_point2point(pred_only_curr_lb, actual)
            metrics[int(lb)] = {
                'f1': p_t[0],
                'precision': p_t[1],
                'recall': p_t[2],
                'TP': p_t[3],
                'TN': p_t[4],
                'FP': p_t[5],
                'FN': p_t[6],
                'ROC/AUC': p_t[7],
                'accuracy': p_t[8],
                'threshold': threshold,
            }

    return metrics, np.array(preds)
