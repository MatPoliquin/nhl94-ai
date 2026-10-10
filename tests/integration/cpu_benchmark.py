"""ROM-dependent CPU reproducibility, real lineup and one-timer checks."""
from nhl94_ai.evaluation.cpu_benchmark import cpu_match


def main():
    fixture = ('classic-v1', 'nordiques-canadiens', 42, 60, 4, 'FILTERED')
    first, repeat = cpu_match(fixture), cpu_match(fixture)
    other = cpu_match(('classic-v1', 'nordiques-canadiens', 43, 60, 4, 'FILTERED'))
    assert first['completed'] and repeat['completed'] and other['completed']
    assert first['lineup'][5]['name'] == 'Patrick Roy'
    for key in ('actions_sha256', 'goals', 'shots', 'one_timers', 'frames'):
        assert first[key] == repeat[key], key
    assert first['actions_sha256'] != other['actions_sha256']
    print('PASS: CPU trials reproduce with the same seed, vary with another, and identify Roy')
    ottawa = cpu_match(('classic-v1', 'senators-penguins', 9003, 300, 4, 'FILTERED'))
    assert ottawa['completed']
    assert ottawa['lineup'][8]['name'] == 'Jamie Baker'
    assert ottawa['one_timers'][1] > 0 and ottawa['decisions'].get('one-timer-pass', 0) > 0
    print('PASS: Ottawa produces a deliberate one-timer against the in-game Pittsburgh CPU')


if __name__ == '__main__':
    main()
