# Tests

Run commands from the repository root in the project's Python environment.

| Directory | Purpose |
| --- | --- |
| `unit/` | Automatically discovered `unittest` cases. Uses synthetic states, mocks, and recorded JSON; no imported game ROM is needed. Some UI checks require the optional display dependencies. |
| `integration/` | Explicit native-emulator checks and longer diagnostic runners. Requires the relevant stable-retro integration, imported ROM, and save states. Some runners open a window or need additional arguments. |
| `support/` | Reusable state builders, fake environments, and resource setup. Contains no test cases and does not start an emulator on import. |
| `fixtures/` | Frozen observation data, input traces, and specific regression scenarios. |

## Running tests

The default suite is the same command used by CI:

```bash
python -m unittest discover -s tests -v
```

Run one module or test while working on a specific area:

```bash
python -m unittest tests.unit.test_classic_offense -v
python -m unittest tests.unit.test_snes_recording -v
python -m unittest tests.unit.test_classic_offense.PassEstimateTests -v
```

Native integration scripts deliberately do not use the `test_*.py` naming
pattern. Run the relevant script explicitly; blanket discovery should not
launch games, run long benchmarks, or require ROMs.

```bash
# Genesis: frozen input traces across game variants and action types.
python -m tests.integration.smoke

# Genesis: native snapshot restore and input replay on both sides.
python -m tests.integration.replay_support

# Genesis and SNES: recording integrity and snapshot seeks.
python -m tests.integration.human_recording
python -m tests.integration.snes_recording

# SNES: seeded reproducibility and paired benchmark behavior.
python -m tests.integration.snes_benchmark
```

Read a runner's module docstring and argument parser before launching it.
`smoke --record` replaces the frozen regression fixtures; ordinary smoke runs
only compare against them.

Repository lint and packaging checks:

```bash
python -m pylint nhl94_ai tests
python -m build --wheel
```

## Shared fixtures

| Module | Contents |
| --- | --- |
| `support/offense.py` | Fresh offensive game states, passing/shooting scenarios, and reception history |
| `support/defense.py` | Defense, goalie, and teammate-pressure states |
| `support/environments.py` | ROM-free observation wrappers, target-control fixtures, away-play setup, and statistics environments |
| `support/teammates.py` | Multi-controller arguments, RAM data, and teammate/multitap environments |
| `support/learned_offense.py` | Deterministic decision ticks and delayed-outcome environments |
| `support/display.py` | Headless pygame setup and debug displays with cleanup registered on the calling test |
| `support/recording.py` | Input-dependent fake emulator state, fake controller diagnostics, and scripted human input |
| `support/memory.py` | Typed overlapping RAM reads and writes |

Import shared fixtures from `tests.support`, rather than importing another
unit-test module or constructing another `TestCase` to borrow its setup.
Builders return fresh mutable states so each test can arrange its own scenario.
Keep scenario-specific assertions and configuration in the test that uses them.

Keep optional display imports inside display helpers, and register cleanup on
the active test case. Native integration scripts may reuse another runner's
domain-specific experiment functions; general-purpose fakes belong in support.

When refactoring, preserve discovered test IDs and assertions, run the unit
suite, and run native checks for any changed integration fixtures. Keep new
tests named `test_*.py` under `unit/` and non-test helpers under `support/`.
