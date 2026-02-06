"""
Statistical analysis utilities.

Provides functions for significance testing and result aggregation.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def paired_ttest(
    group1: List[float],
    group2: List[float],
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Perform paired t-test between two groups.
    
    Args:
        group1: First group values
        group2: Second group values
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        Dict with 't_statistic', 'p_value', 'significant' (p<0.05)
    """
    if len(group1) != len(group2):
        raise ValueError("Groups must have same length for paired test")
    
    if len(group1) < 3:
        return {"t_statistic": 0.0, "p_value": 1.0, "significant": False}
    
    t_stat, p_val = stats.ttest_rel(group1, group2, alternative=alternative)
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "significant": p_val < 0.05,
    }


def wilcoxon_test(
    group1: List[float],
    group2: List[float],
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Perform Wilcoxon signed-rank test (non-parametric).
    
    Better for small samples or non-normal distributions.
    """
    if len(group1) != len(group2):
        raise ValueError("Groups must have same length")
    
    if len(group1) < 3:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}
    
    stat, p_val = stats.wilcoxon(group1, group2, alternative=alternative)
    
    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "significant": p_val < 0.05,
    }


def compute_effect_size(
    group1: List[float],
    group2: List[float],
) -> Dict[str, float]:
    """
    Compute Cohen's d effect size.
    
    NOTE: Thresholds (0.2/0.5/0.8) are from Cohen (1988) but are guidelines,
    not universal standards. We report raw values; interpretation depends on context.
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        cohens_d = 0.0
    else:
        cohens_d = (mean1 - mean2) / pooled_std
    
    return {
        "cohens_d": float(cohens_d),
        "mean_diff": float(mean1 - mean2),
    }


def bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Multiply each p-value by the number of tests to control
    family-wise error rate at α=0.05 across all comparisons.
    
    Args:
        p_values: List of uncorrected p-values
        
    Returns:
        List of corrected p-values (capped at 1.0)
    """
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def mann_whitney_test(
    group1: List[float],
    group2: List[float],
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Perform Mann-Whitney U test (non-parametric, independent samples).
    
    Use this when samples are independent (unlike Wilcoxon which requires paired).
    
    Args:
        group1: First group values
        group2: Second group values  
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        Dict with 'statistic', 'p_value', 'significant'
    """
    if len(group1) < 2 or len(group2) < 2:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}
    
    stat, p_val = stats.mannwhitneyu(group1, group2, alternative=alternative)
    
    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "significant": p_val < 0.05,
    }


def compute_all_statistics(
    euclidean_scores: List[float],
    hyperbolic_scores: List[float],
    metric_name: str = "spearman_rho",
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for Euclidean vs Hyperbolic comparison.
    
    Includes:
    - Descriptive statistics (mean, std)
    - Bootstrap 95% CI for both groups
    - Bootstrap 95% CI for the difference
    - Paired t-test
    - Wilcoxon signed-rank test
    - Mann-Whitney U test
    - Cohen's d effect size
    
    Args:
        euclidean_scores: Euclidean probe scores (one per layer/fold)
        hyperbolic_scores: Hyperbolic probe scores (one per layer/fold)
        metric_name: Name of metric being compared
        n_bootstrap: Number of bootstrap iterations
        
    Returns:
        Comprehensive statistics dict
    """
    euclidean = np.array(euclidean_scores)
    hyperbolic = np.array(hyperbolic_scores)
    
    # Descriptive stats
    desc = {
        "metric": metric_name,
        "euclidean_mean": float(np.mean(euclidean)),
        "euclidean_std": float(np.std(euclidean)),
        "hyperbolic_mean": float(np.mean(hyperbolic)),
        "hyperbolic_std": float(np.std(hyperbolic)),
        "n_samples": len(euclidean),
        "mean_improvement": float(np.mean(hyperbolic) - np.mean(euclidean)),
        "percent_improvement": float(
            100 * (np.mean(hyperbolic) - np.mean(euclidean)) / (np.mean(euclidean) + 1e-10)
        ),
    }
    
    # Bootstrap CIs
    desc["euclidean_ci_95"] = bootstrap_ci(euclidean_scores, n_bootstrap=n_bootstrap)
    desc["hyperbolic_ci_95"] = bootstrap_ci(hyperbolic_scores, n_bootstrap=n_bootstrap)
    
    # Bootstrap CI for difference
    diffs = []
    for _ in range(n_bootstrap):
        s1 = np.random.choice(euclidean, size=len(euclidean), replace=True)
        s2 = np.random.choice(hyperbolic, size=len(hyperbolic), replace=True)
        diffs.append(np.mean(s2) - np.mean(s1))
    desc["diff_ci_95"] = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
    
    # Statistical tests (hyperbolic > euclidean)
    tests = {}
    
    if len(euclidean) == len(hyperbolic) and len(euclidean) >= 3:
        tests["paired_ttest"] = paired_ttest(hyperbolic_scores, euclidean_scores, alternative="greater")
        tests["wilcoxon"] = wilcoxon_test(hyperbolic_scores, euclidean_scores, alternative="greater")
    
    tests["mann_whitney"] = mann_whitney_test(hyperbolic_scores, euclidean_scores, alternative="greater")
    
    # Effect size
    effect = compute_effect_size(hyperbolic_scores, euclidean_scores)
    
    return {
        "descriptive": desc,
        "tests": tests,
        "effect_size": effect,
        "hyperbolic_better": desc["hyperbolic_mean"] > desc["euclidean_mean"],
        "significant": any(t.get("significant", False) for t in tests.values()),
    }


def bootstrap_ci(
    data: List[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    statistic: str = "mean",
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval.
    
    Args:
        data: Sample data
        n_bootstrap: Number of bootstrap iterations
        alpha: Significance level (default 0.05 for 95% CI)
        statistic: "mean" or "median"
        
    Returns:
        (lower_bound, upper_bound) tuple
    """
    data = np.array(data)
    n = len(data)
    
    if n == 0:
        return (0.0, 0.0)
    
    stat_func = np.mean if statistic == "mean" else np.median
    
    # Bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(stat_func(sample))
    
    # Percentile method
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return (float(lower), float(upper))


def compare_probe_types(
    results: List[Dict[str, Any]],
    metric: str = "spearman_rho",
) -> Dict[str, Any]:
    """
    Compare Euclidean vs Hyperbolic probes across all results.
    
    Args:
        results: List of experiment result dicts
        metric: Metric to compare
        
    Returns:
        Full comparison with statistics
    """
    euclidean = [r[metric] for r in results if r["probe_type"] == "euclidean"]
    hyperbolic = [r[metric] for r in results if r["probe_type"] == "hyperbolic"]
    
    # Descriptive stats
    desc = {
        "euclidean_mean": float(np.mean(euclidean)),
        "euclidean_std": float(np.std(euclidean)),
        "hyperbolic_mean": float(np.mean(hyperbolic)),
        "hyperbolic_std": float(np.std(hyperbolic)),
        "n_samples": len(euclidean),
    }
    
    # Statistical tests
    ttest = paired_ttest(hyperbolic, euclidean, alternative="greater")
    wilcox = wilcoxon_test(hyperbolic, euclidean, alternative="greater")
    effect = compute_effect_size(hyperbolic, euclidean)
    
    return {
        "descriptive": desc,
        "ttest": ttest,
        "wilcoxon": wilcox,
        "effect_size": effect,
        "hyperbolic_better": desc["hyperbolic_mean"] > desc["euclidean_mean"],
        "significant": ttest["significant"] or wilcox["significant"],
    }


def aggregate_by_layer(
    results: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """
    Aggregate results by layer.
    """
    by_layer = {}
    
    for r in results:
        layer = r["layer"]
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append(r)
    
    aggregated = {}
    for layer, layer_results in by_layer.items():
        euclidean = [r for r in layer_results if r["probe_type"] == "euclidean"]
        hyperbolic = [r for r in layer_results if r["probe_type"] == "hyperbolic"]
        
        aggregated[layer] = {
            "euclidean_rho": np.mean([r["spearman_rho"] for r in euclidean]) if euclidean else 0,
            "hyperbolic_rho": np.mean([r["spearman_rho"] for r in hyperbolic]) if hyperbolic else 0,
            "improvement": (
                np.mean([r["spearman_rho"] for r in hyperbolic]) - 
                np.mean([r["spearman_rho"] for r in euclidean])
            ) if euclidean and hyperbolic else 0,
        }
    
    return aggregated


def find_best_layer(
    results: List[Dict[str, Any]],
    probe_type: str = "hyperbolic",
    metric: str = "spearman_rho",
) -> Dict[str, Any]:
    """
    Find the best performing layer.
    """
    filtered = [r for r in results if r["probe_type"] == probe_type]
    
    if not filtered:
        return {"layer": -1, "value": 0.0}
    
    by_layer = {}
    for r in filtered:
        layer = r["layer"]
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append(r[metric])
    
    best_layer = max(by_layer.keys(), key=lambda l: np.mean(by_layer[l]))
    
    return {
        "layer": best_layer,
        "value": float(np.mean(by_layer[best_layer])),
        "all_layers": {l: float(np.mean(v)) for l, v in by_layer.items()},
    }
