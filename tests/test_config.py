"""configs/paper_v1.yaml parity, strict loading, overrides, and the optional weight-sum normalisation."""
from dataclasses import fields
import json
from pathlib import Path

import pytest
import torch
import yaml

from hifidc.config import Config, load_config, parse_overrides
from hifidc.core import data_terms

PAPER_V1 = Path(__file__).resolve().parent.parent / 'configs' / 'paper_v1.yaml'


def test_paper_v1_equals_defaults():
    assert load_config(PAPER_V1) == Config()


def test_every_field_has_provenance():
    document = yaml.safe_load(PAPER_V1.read_text())
    names = {f.name for f in fields(Config)}
    assert set(document['values']) == names
    assert set(document['provenance']) == names
    for name, source in document['provenance'].items():
        assert source == 'paper' or source.startswith('reconstruction: '), name


def test_paper_sourced_values():
    document = yaml.safe_load(PAPER_V1.read_text())
    for name in ('steps', 'normal_weight', 'lambda_l1', 'lsq_interval', 'ema_beta', 'edge_tau',
                 'stability_threshold', 'bilateral_sigma_space', 'bilateral_sigma_range'):
        assert document['provenance'][name] == 'paper', name
    assert document['values']['edge_tau'] == 0.20 and document['values']['steps'] == 50


def test_unknown_key_rejected(tmp_path):
    document = yaml.safe_load(PAPER_V1.read_text())
    document['values']['not_a_field'] = 1
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match='not_a_field'):
        load_config(path)


def test_overrides_apply_and_validate():
    assert load_config(PAPER_V1, {'steps': 20}).steps == 20
    with pytest.raises(ValueError):
        load_config(PAPER_V1, {'learning_rate': 0.0})  # LR must be positive: the no-guidance control uses 1e-12
    with pytest.raises(ValueError, match='bogus'):
        load_config(PAPER_V1, {'bogus': 1})


def test_values_coerced_to_field_types(tmp_path):
    # YAML 1.1 reads 1e-12 (no decimal point) as a string; overrides arrive as strings from the CLI.
    config = load_config(PAPER_V1, parse_overrides(['learning_rate=1e-12', 'steps=6', 'use_normals=false',
                                                    'normal_schedule=constant']))
    assert config.learning_rate == 1e-12 and isinstance(config.learning_rate, float)
    assert config.steps == 6 and isinstance(config.steps, int)
    assert config.use_normals is False and config.normal_schedule == 'constant'
    path = tmp_path / 'sci.yaml'
    path.write_text('ridge: 1e-6\nsteps: 20\n')
    assert load_config(path).ridge == 1e-6
    with pytest.raises(ValueError, match='steps'):
        load_config(PAPER_V1, parse_overrides(['steps=abc']))
    with pytest.raises(ValueError, match='steps'):
        load_config(PAPER_V1, {'steps': 2.5})
    with pytest.raises(ValueError, match='key=value'):
        parse_overrides(['steps'])


def test_flat_json_legacy_format(tmp_path):
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps({'steps': 30, 'normal_weight': 0.05}))
    config = load_config(path)
    assert config.steps == 30 and config.normal_weight == 0.05 and config.edge_tau == 0.20


def test_weight_sum_normalization():
    p = torch.tensor([1., 3., 5., 7.]).reshape(1, 1, 2, 2)
    t, w = torch.ones_like(p), torch.tensor([1., .5, .25, .1]).reshape_as(p)
    r, sw = p - t, w.sum()
    q, l1 = data_terms(p, t, t.bool(), w, Config(data_normalization='weight_sum'))
    assert abs(float(q) - float((w * r * r).sum() / sw - .5 * ((w * r).sum() / sw) ** 2)) < 1e-6
    assert abs(float(l1) - float(.5 * (w * r.abs()).sum() / sw)) < 1e-6
    q_count, l1_count = data_terms(p, t, t.bool(), w, Config())
    assert abs(float(q_count) - float((w * r * r).sum() / 4 - (w * r).sum() ** 2 / 32)) < 1e-6
    with pytest.raises(ValueError):
        Config(data_normalization='median').validate()
