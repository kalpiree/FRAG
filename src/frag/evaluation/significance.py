from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from typing import Any


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    maximum_iterations = 300
    epsilon = 3e-14
    tiny = sys.float_info.min / epsilon
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        doubled = 2 * iteration
        coefficient = (
            iteration
            * (b - iteration)
            * x
            / ((qam + doubled) * (a + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c
        coefficient = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + doubled) * (qap + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise ArithmeticError("incomplete beta evaluation did not converge")


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if a <= 0.0 or b <= 0.0:
        raise ValueError("beta parameters must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        value = front * _beta_continued_fraction(a, b, x) / a
    else:
        value = 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b
    return min(1.0, max(0.0, value))


def _student_t_two_sided_p(statistic: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError("degrees of freedom must be positive")
    if math.isinf(statistic):
        return 0.0
    x = degrees_of_freedom / (degrees_of_freedom + statistic * statistic)
    return _regularized_incomplete_beta(degrees_of_freedom / 2.0, 0.5, x)


def _student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if value == 0.0:
        return 0.5
    probability = _student_t_two_sided_p(value, degrees_of_freedom) / 2.0
    return 1.0 - probability if value > 0.0 else probability


def _student_t_quantile(probability: float, degrees_of_freedom: int) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be between zero and one")
    if probability == 0.5:
        return 0.0
    if probability < 0.5:
        return -_student_t_quantile(1.0 - probability, degrees_of_freedom)
    lower = 0.0
    upper = 1.0
    while _student_t_cdf(upper, degrees_of_freedom) < probability:
        upper *= 2.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if _student_t_cdf(midpoint, degrees_of_freedom) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def paired_t_test(
    first: Sequence[float],
    second: Sequence[float],
) -> dict[str, float | int]:
    try:
        first_values = [float(value) for value in first]
        second_values = [float(value) for value in second]
    except (TypeError, ValueError) as error:
        raise ValueError("paired samples must be one-dimensional numeric values") from error
    if len(first_values) != len(second_values):
        raise ValueError("paired samples must have equal shape")
    if len(first_values) < 2:
        raise ValueError("paired test requires at least two pairs")
    if not all(math.isfinite(value) for value in first_values + second_values):
        raise ValueError("paired samples must be finite")
    differences = [
        first_value - second_value
        for first_value, second_value in zip(first_values, second_values, strict=True)
    ]
    sample_size = len(differences)
    degrees_of_freedom = sample_size - 1
    mean_difference = sum(differences) / sample_size
    squared_deviations = sum(
        (difference - mean_difference) ** 2 for difference in differences
    )
    standard_deviation = math.sqrt(squared_deviations / degrees_of_freedom)
    if standard_deviation == 0.0:
        statistic = 0.0 if mean_difference == 0.0 else math.copysign(math.inf, mean_difference)
    else:
        statistic = mean_difference / (standard_deviation / math.sqrt(sample_size))
    p_value = _student_t_two_sided_p(statistic, degrees_of_freedom)
    return {
        "n": sample_size,
        "degrees_of_freedom": degrees_of_freedom,
        "mean_difference": mean_difference,
        "statistic": statistic,
        "p_value": p_value,
    }


def paired_t_test_by_user(
    first: Mapping[Any, float], second: Mapping[Any, float]
) -> dict[str, float | int]:
    if set(first) != set(second):
        raise ValueError("paired user scores must contain identical users")
    users = tuple(first)
    return paired_t_test(
        [first[user] for user in users], [second[user] for user in users]
    )


def mean_confidence_interval(
    values: Sequence[Any],
    confidence: float = 0.95,
    axis: int = 0,
) -> dict[str, Any]:
    if axis != 0:
        raise ValueError("only axis zero is supported")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    samples = list(values)
    if len(samples) < 2:
        raise ValueError("confidence interval requires at least two samples")
    scalar = isinstance(samples[0], int | float)
    rows = (
        [[float(sample)] for sample in samples]
        if scalar
        else [list(map(float, sample)) for sample in samples]
    )
    if not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("sample rows must have equal nonzero length")
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError("values must be finite")
    sample_size = len(rows)
    means = [
        sum(row[index] for row in rows) / sample_size
        for index in range(len(rows[0]))
    ]
    standard_errors = [
        math.sqrt(
            sum((row[index] - means[index]) ** 2 for row in rows)
            / (sample_size - 1)
        )
        / math.sqrt(sample_size)
        for index in range(len(rows[0]))
    ]
    critical = _student_t_quantile(
        (1.0 + confidence) / 2.0, sample_size - 1
    )
    margins = [critical * error for error in standard_errors]
    lower = [mean - margin for mean, margin in zip(means, margins, strict=True)]
    upper = [mean + margin for mean, margin in zip(means, margins, strict=True)]
    mean_result: Any = means[0] if scalar else means
    lower_result: Any = lower[0] if scalar else lower
    upper_result: Any = upper[0] if scalar else upper
    return {
        "n": sample_size,
        "confidence": confidence,
        "mean": mean_result,
        "lower": lower_result,
        "upper": upper_result,
    }
