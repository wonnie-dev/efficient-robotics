"""Categorical calibration, entropy, and Bayesian filtering utilities."""

import math


def normalize(distribution: dict[str, float], floor: float = 1e-12) -> dict[str, float]:
    """Return a proper categorical distribution without exact zero entries."""
    clipped = {key: max(floor, float(value)) for key, value in distribution.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        raise ValueError("Categorical distribution has no probability mass")
    return {key: value / total for key, value in clipped.items()}


def entropy(distribution: dict[str, float]) -> float:
    normalized = normalize(distribution)
    return -sum(p * math.log(p) for p in normalized.values())


def softmax_temperature(logits: list[float], temperature: float) -> list[float]:
    if temperature <= 0.0:
        raise ValueError("Temperature must be positive")
    scaled = [value / temperature for value in logits]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def negative_log_likelihood(
    logits_batch: list[list[float]], labels: list[int], temperature: float
) -> float:
    if len(logits_batch) != len(labels) or not labels:
        raise ValueError("Calibration logits and labels must be non-empty and aligned")
    losses = []
    for logits, label in zip(logits_batch, labels):
        probabilities = softmax_temperature(logits, temperature)
        if not 0 <= label < len(probabilities):
            raise ValueError("Calibration label is outside the logit dimension")
        losses.append(-math.log(max(1e-12, probabilities[label])))
    return sum(losses) / len(losses)


def fit_temperature_grid(
    logits_batch: list[list[float]],
    labels: list[int],
    minimum: float = 0.25,
    maximum: float = 5.0,
    steps: int = 191,
) -> dict:
    """Choose the temperature with the lowest calibration-set log loss."""
    if steps < 2 or minimum <= 0.0 or maximum <= minimum:
        raise ValueError("Invalid temperature search range")
    candidates = [
        minimum + index * (maximum - minimum) / (steps - 1)
        for index in range(steps)
    ]
    best = min(
        candidates,
        key=lambda value: negative_log_likelihood(logits_batch, labels, value),
    )
    return {
        "temperature": best,
        "negative_log_likelihood": negative_log_likelihood(
            logits_batch, labels, best
        ),
        "method": "temperature_scaling_grid_search",
        "calibrated": True,
    }


def bayesian_update(
    prior: dict[str, float],
    likelihood_by_hypothesis: dict[str, float],
) -> dict[str, float]:
    """Apply one categorical Bayes update and normalize the posterior."""
    if prior.keys() != likelihood_by_hypothesis.keys():
        raise ValueError("Prior and likelihood hypotheses must match")
    # The evidence normalizer is shared across hypotheses, so it is applied
    # once by normalize after multiplying prior mass by the likelihood.
    posterior = {
        hypothesis: prior[hypothesis] * likelihood_by_hypothesis[hypothesis]
        for hypothesis in prior
    }
    return normalize(posterior)


def binary_detection_likelihood(
    detection_probability: dict[str, float], detected: bool
) -> dict[str, float]:
    """Select P(detection | hypothesis) or its complement for each state."""
    return {
        hypothesis: probability if detected else 1.0 - probability
        for hypothesis, probability in detection_probability.items()
    }
