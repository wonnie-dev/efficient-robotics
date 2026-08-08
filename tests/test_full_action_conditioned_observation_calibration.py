from scripts.run_full_action_conditioned_observation_calibration import (
    fit_likelihood,
    leave_one_episode_out,
    posterior,
)


def test_likelihood_is_normalized_and_observation_changes_belief():
    examples = [
        {"episode_key": "i0", "seed": 0, "state": "inside", "observation": "target_detected"},
        {"episode_key": "i1", "seed": 1, "state": "inside", "observation": "target_detected"},
        {"episode_key": "o0", "seed": 2, "state": "outside", "observation": "empty_container"},
        {"episode_key": "o1", "seed": 3, "state": "outside", "observation": "empty_container"},
    ]
    states = ["inside", "outside"]
    observations = ["target_detected", "empty_container"]
    likelihood = fit_likelihood(examples, states, observations, 1.0)
    assert all(abs(sum(row.values()) - 1.0) < 1e-12 for row in likelihood.values())
    target_belief = posterior("target_detected", likelihood, states)
    empty_belief = posterior("empty_container", likelihood, states)
    assert target_belief["inside"] > target_belief["outside"]
    assert empty_belief["outside"] > empty_belief["inside"]


def test_leave_one_episode_out_is_episode_disjoint():
    examples = [
        {"episode_key": "i0", "seed": 0, "state": "inside", "observation": "target_detected"},
        {"episode_key": "i1", "seed": 1, "state": "inside", "observation": "target_detected"},
        {"episode_key": "o0", "seed": 2, "state": "outside", "observation": "empty_container"},
        {"episode_key": "o1", "seed": 3, "state": "outside", "observation": "empty_container"},
    ]
    result = leave_one_episode_out(
        examples,
        ["inside", "outside"],
        ["target_detected", "empty_container"],
        1.0,
    )
    assert result["fold_count"] == 4
    assert result["correct_count"] == 4
    assert result["accuracy"] == 1.0
