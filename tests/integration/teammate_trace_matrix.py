"""Rerun the retained sixteen live fixtures without overwriting their traces."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
from pathlib import Path

from nhl94_ai.evaluation.teammates import play_match, source_hash
from nhl94_ai.evaluation.teammate_trace_compare import compare_fixture_matrix


def measure(job):
    output, name, settings = job
    with (output/(name+'.log')).open('x') as log, redirect_stdout(log), redirect_stderr(log):
        result = play_match(**settings, trace=output/(name+'.jsonl'))
    assert result['completed']
    return name, {'settings': settings, 'result': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    before = json.loads(args.reference.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    for path, digest in before['assets'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    source = source_hash()
    with ProcessPoolExecutor(max_workers=6) as pool:
        cases = dict(pool.map(measure, [(args.output, name, case['settings']) for name, case in before['teammates'].items()]))
    assert source_hash() == source
    report = args.output/'report.json'
    report.write_text(json.dumps({'teammates': cases, 'source_sha256': source, 'assets': before['assets']}, indent=2)+'\n')
    comparisons = compare_fixture_matrix(args.reference, report)
    (args.output/'comparison.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    print({name: row['first_action_difference']['frame'] if row['first_action_difference'] else None for name, row in comparisons.items()})


if __name__ == '__main__':
    main()
