"""Typed, overlapping RAM writes for emulator-free full-team reset tests."""
import struct


_FORMATS = {'>i4': '>i', '>u4': '>I', '>i2': '>h', '>u2': '>H', '|u1': '>B', '|i1': '>b'}


class FixtureMemory:
    def __init__(self, info):
        self.info = info
        self.buffer = bytearray(65536)
        self.fields = {
            'period': (0xFFC466, '>u2'), 'time': (0xFFC466, '>u4'),
            'puck_owner': (0xFFB7AA, '>i2'), 'shot_player': (0xFFBED8, '>i2'),
            'sflags': (0xFFC2EC, '>u2'), 'sflags2': (0xFFC2EE, '>u2'),
            'pass_target': (0xFFBEE0, '>i2'), 'last_puck_player': (0xFFBEDA, '>i2'),
        }
        for slot in (*range(12), 14):
            if slot == 14:
                prefix = 'puck_'
            elif slot % 6 == 5:
                prefix = f'g{slot // 6 + 1}_'
            else:
                prefix = f'p{slot // 6 + 1}_' + (f'{slot % 6 + 1}_' if slot % 6 else '')
            base = 0xFFB04A + slot * 0x80
            self._write(base + 0x4A, '>u2', 5 if slot == 14 else 8)
            self._write(base + 0x4C, '>u2', 5 if slot == 14 else 4)
            for name, offset, kind in (
                ('x', 0, '>i2'), ('y', 0x14, '>i2'), ('z', 0x18, '>i2'),
                ('vel_x', 0x28, '|i1'), ('vel_y', 0x2A, '|i1'), ('vel_z', 0x2C, '|i1'),
                ('ori', 0x55, '|u1'), ('anim', 0x58, '>u2'),
                ('anim_frame', 0x5A, '>u2'), ('state_flags', 0x64, '|u1'),
            ):
                self.fields[prefix + name] = base + offset, kind
            if slot < 12:
                self._write(base + 0x34, '>i2', (5, 3, 4, 2, 1, 0)[slot % 6])
                self._write(base + 0x62, '|u1', (0x80 if slot < 6 else 0x40) | (8 if slot == 1 else 0))
                self._write(base + 0x6A, '|u1', 12)
        self._write(0xFFC320, '>i2', 1)
        self._write(0xFFC322, '>i2', -1)
        self._write(0xFFC328, '>u2', 1)
        for name, (address, kind) in self.fields.items():
            if name in info:
                self._write(address, kind, info[name])

    def _write(self, address, kind, value):
        struct.pack_into(_FORMATS[kind], self.buffer, address & 0xFFFF, value)

    def extract(self, address, kind):
        return struct.unpack_from(_FORMATS[kind], self.buffer, address & 0xFFFF)[0]

    def assign(self, address, kind, value):
        self._write(address, kind, value)
        end = address + struct.calcsize(_FORMATS[kind])
        for name, (field_address, field_kind) in self.fields.items():
            if address < field_address + struct.calcsize(_FORMATS[field_kind]) and field_address < end:
                self.info[name] = self.extract(field_address, field_kind)
