"""Fail early on CSV files that cannot meaningfully enter the lakehouse."""
import json

import pytest

from lakehouse.contracts import validate_contracts


@pytest.mark.parametrize('csv_text, reason', [
    ('id,value\n', 'empty_source'),
    ('id,value,value\n1,2,3\n', 'duplicate_column:value'),
    ('id,value\n1,2,3\n', 'malformed_row:2'),
    ('id,value\n1\n', 'malformed_row:2'),
])
def test_unusable_csv_fails_with_source_name_and_retains_report(tmp_path, csv_text, reason):
    source = tmp_path / 'campaign.csv'
    source.write_text(csv_text)
    contract = tmp_path / 'contract.json'
    contract.write_text(json.dumps({'version': 'test', 'sources': [{
        'name': 'campaign_example', 'path': str(source), 'required_columns': ['id', 'value'],
    }]}))
    report = tmp_path / 'report.json'
    with pytest.raises(ValueError, match='campaign_example'):
        validate_contracts(contract, report)
    result = json.loads(report.read_text())
    assert result['status'] == 'fail'
    assert reason in result['sources'][0]['fatal']
