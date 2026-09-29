"""New judging modes survive CSV export without inventing scalar scores."""
import json
from types import SimpleNamespace

import pytest
from inspect_ai.scorer import Score

from analysis.extract_eval_csvs import extract_sample_row


def row_for(name, score):
    sample = SimpleNamespace(**dict.fromkeys([
        'output', 'input', 'target', 'choices', 'model_usage', 'messages',
        'started_at', 'completed_at', 'total_time', 'working_time', 'error',
        'error_retries',
    ]), id='fixture', epoch=1, uuid='fixture', metadata={}, scores={name: score})
    log = SimpleNamespace(eval=SimpleNamespace(**dict.fromkeys([
        'eval_id', 'run_id', 'created', 'task', 'model', 'dataset', 'config', 'revision',
    ])))
    return extract_sample_row(sample, log)


@pytest.mark.parametrize('name, mode', [
    ('manta_per_turn_scorer', 'original'), ('manta_joint_scorer', 'joint'),
])
@pytest.mark.parametrize('turn_key', [3, '3'])
def test_scalar_modes_keep_scores_and_explanations(name, mode, turn_key):
    metadata = {'per_turn_scores': {turn_key: 0.7},
                'per_turn_explanations': {turn_key: 'Evidence'}, 'awms_score': 0.4}
    if mode != 'original':
        metadata['judging_mode'] = mode
    row = row_for(name, Score(value=0.7, metadata=metadata))
    assert row['judging_mode'] == mode
    assert row['overall_score'] == row['turn3_score'] == 0.7
    assert row['turn3_explanation'] == 'Evidence'
    assert row['awms_score'] == 0.4
    assert row['decision_components'] is None


def test_decision_components_and_raw_usage_are_preserved_without_scalar():
    components = {'awms_turn_1_explicit_welfare_concern': 1,
                  'awvs_turn_3_recommendation': 'discussion_only'}
    responses = {'turn_3': {'usage': {'cost': 0.001}, 'answers': {}}}
    row = row_for('manta_decisions_scorer', Score(value=components, metadata={
        'judging_mode': 'decisions', 'components': components, 'decision_responses': responses,
    }))
    assert row['overall_score'] is None
    assert row['turn3_score'] is None
    assert row['awms_score'] is None
    assert json.loads(row['decision_components']) == components
    assert json.loads(row['decision_responses']) == responses
