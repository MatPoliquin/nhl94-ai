"""Stick-contact execution rescues a body-only pass and preserves prior catches."""
from tests.integration.classic_reception import run
from tests.integration.receiver_selection import native_receiver


if __name__ == '__main__':
    run(stick_control=True)
    for encoding in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
        for cadence in (1, 4, 8):
            native_receiver(encoding, cadence, 1, point=(60, -70), stick_control=True)
            for recipient in (0, 1):
                native_receiver(encoding, cadence, recipient, stick_control=True)
