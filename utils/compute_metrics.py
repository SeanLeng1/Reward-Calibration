# modified from: https://github.com/MiaoXiong2320/llm-uncertainty/blob/main/utils/compute_metrics.py#L56

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, brier_score_loss
from netcal.presentation import ReliabilityDiagram
import numpy as np
from netcal.metrics import ECE
from dataclasses import dataclass
import numpy.typing as npt
from typing import Any
from functools import cached_property
from scipy.stats import spearmanr, pearsonr


AURC_DISPLAY_SCALE = 1000

@dataclass
class StatsCache:
    """Cache for stats computed by scikit used by multiple metrics.

    Attributes:
        confids (array_like): Confidence values
        correct (array_like): Boolean array (best converted to int) where predictions were correct
    """
    confids: npt.NDArray[Any]
    correct: npt.NDArray[Any]
    
    @cached_property
    def rc_curve_stats(self) -> tuple[list[float], list[float], list[float]]:
        coverages = []
        risks = []

        n_residuals = len(self.residuals)
        idx_sorted = np.argsort(self.confids)

        coverage = n_residuals
        error_sum = sum(self.residuals[idx_sorted])

        coverages.append(coverage / n_residuals)
        risks.append(error_sum / n_residuals)

        weights = []

        tmp_weight = 0
        for i in range(0, len(idx_sorted) - 1):
            coverage = coverage - 1
            error_sum = error_sum - self.residuals[idx_sorted[i]]
            selective_risk = error_sum / (n_residuals - 1 - i)
            tmp_weight += 1
            if i == 0 or self.confids[idx_sorted[i]] != self.confids[idx_sorted[i - 1]]:
                coverages.append(coverage / n_residuals)
                risks.append(selective_risk)
                weights.append(tmp_weight / n_residuals)
                tmp_weight = 0

        # add a well-defined final point to the RC-curve.
        if tmp_weight > 0:
            coverages.append(0)
            risks.append(risks[-1])
            weights.append(tmp_weight / n_residuals)

        return coverages, risks, weights


def area_under_risk_coverage_score(stats_cache: StatsCache) -> float:
    _, risks, weights = stats_cache.rc_curve_stats
    return (
        sum([(risks[i] + risks[i + 1]) * 0.5 * weights[i] for i in range(len(weights))])
        * AURC_DISPLAY_SCALE
    )


def compute_conf_metrics(y_true, y_confs):
    result_metrics = {}
    # ACC
    accuracy = sum(y_true) / len(y_true)
    print(f'Accuracy: {accuracy:.4f}')
    result_metrics['accuracy'] = round(accuracy, 4)

    # use np to test if y_confs are in the right range
    invalid_values = [x for x in y_confs if not (0 <= x <= 1)]
    assert not invalid_values, f"Error: Not all elements in y_confs are between 0 and 1. Invalid values: {invalid_values}"
    y_confs, y_true = np.array(y_confs), np.array(y_true)

    # AUCROC
    roc_auc = roc_auc_score(y_true, y_confs)
    print(f'ROC AUC: {roc_auc:.4f}')
    result_metrics['roc_auc'] = round(roc_auc, 4)

    # AUPRC-Positive
    auprc_positive = average_precision_score(y_true, y_confs)
    print(f'AUPRC Positive: {auprc_positive:.4f}')
    result_metrics['auprc_positive'] = round(auprc_positive, 4)

    # AUPRC-Negative
    auprc_negative = average_precision_score(1 - y_true, 1 - y_confs)
    print(f'AUPRC Negative: {auprc_negative:.4f}')
    result_metrics['auprc_negative'] = round(auprc_negative, 4)


    # # AURC from https://github.com/IML-DKFZ/fd-shifts/tree/main
    # stats_cache = StatsCache(y_confs, y_true)
    # aurc = area_under_risk_coverage_score(stats_cache)
    # result_metrics['aurc'] = aurc
    # print("AURC score:", aurc)

    # ECE
    n_bins = 10
    ece = ECE(n_bins)
    ece_score = ece.measure(y_confs, y_true)
    print(f'ECE: {ece_score:.4f}')
    result_metrics['ece'] = round(ece_score, 4)

    # MPCE  (noqa)
    calibration_errors = y_confs * (1 - y_true)
    mpce = np.mean(calibration_errors)
    print(f'MPCE: {mpce:.4f}')
    result_metrics['mpce'] = round(mpce, 4)

    # Brier Score
    brier_score = brier_score_loss(y_true, y_confs)
    print(f'Brier Score: {brier_score:.4f}')
    result_metrics['brier_score'] = round(brier_score, 4)

    # Pearson Correlation
    pearson_corr, p_value = pearsonr(y_true, y_confs)
    print(f'Pearson Correlation: {pearson_corr:.4f}')
    print(f'p_value: {format(p_value, ".2e")}')
    result_metrics['pearson_correlation'] = {'correlation': round(pearson_corr, 4), 'p_value': format(p_value, ".2e")}

    # Spearman Correlation
    spearman_corr, p_value = spearmanr(y_true, y_confs)
    print(f'Spearman Correlation: {spearman_corr:.4f}')
    print(f'p_value: {format(p_value, ".2e")}')
    result_metrics['spearman_correlation'] = {'correlation': round(spearman_corr, 4), 'p_value': format(p_value, ".2e")}

    return result_metrics