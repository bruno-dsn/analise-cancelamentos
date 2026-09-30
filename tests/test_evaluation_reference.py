import numpy as np
import pytest
from src.model import calcular_metricas


def test_majority_accuracy_does_not_hide_zero_recall():
    y = np.array([0] * 9 + [1])
    metrics, matrix = calcular_metricas(y, np.full(10, .1), .2)
    assert metrics['acuracia'] == pytest.approx(.9)
    assert metrics['recall'] == 0
    assert metrics['average_precision'] == pytest.approx(.1)
    assert metrics['roc_auc'] == pytest.approx(.5)
    assert matrix == [[9, 0], [1, 0]]
