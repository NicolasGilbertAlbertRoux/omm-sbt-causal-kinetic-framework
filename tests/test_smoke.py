from omm_sbt.core.evolution import run_evolution
from omm_sbt.core.fields import initialize_state
from omm_sbt.models.parameters import ModelParameters
from omm_sbt.analysis.metrics import sector_scores


def test_smoke_evolution_and_scores():
    state0 = initialize_state(size=32, family="mixed", seed=123)
    states, _ = run_evolution(state0, ModelParameters(), steps=5)
    scores = sector_scores(states)
    assert "core_score" in scores
    assert 0.0 <= scores["core_score"] <= 1.0
