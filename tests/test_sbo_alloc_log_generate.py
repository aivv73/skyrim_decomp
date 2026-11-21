"""BNet-runtime SBO binder, custom-allocator factory, buffer-clear, and guarded-log wrappers in an original synthetic PE image."""
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import check_code_rules
import contrib_screen
import lint_reconstruction
from pecoff import PeImage
import sbo_alloc_log_generate as generator
from synthetic_pecoff import assemble_pe


def _rel32(pos, length, target):
    return struct.pack('<i', target - (pos + length))


def _call_rel32(pos, target):
    return bytes([0xe8]) + _rel32(pos, 5, target)


def _jmp_rel32(pos, target):
    return bytes([0xe9]) + _rel32(pos, 5, target)


def _lea_rip(pos, modrm, target):
    return bytes([0x48, 0x8d, modrm]) + _rel32(pos, 7, target)


def _mov_r32_rip(pos, modrm, target):
    return bytes([0x8b, modrm]) + _rel32(pos, 6, target)


def _mov_rax_rip(pos, target):
    return bytes([0x48, 0x8b, 0x05]) + _rel32(pos, 7, target)


def _call_rip_ind(pos, target):
    return bytes([0xff, 0x15]) + _rel32(pos, 6, target)


def _cmp_rip_qword_0(pos, target):
    return bytes([0x48, 0x83, 0x3d]) + _rel32(pos, 8, target) + bytes([0x00])


def _guarded_log_u64(pos, ch, f_filter, f_cen, f_verb, f_con, f_sen, f_sink, fmt, arg):
    b = bytearray()
    b += _mov_r32_rip(pos + len(b), 0x0d, ch)           # mov ecx, [ch]
    b += bytes.fromhex('33 d2')                         # xor edx, edx
    b += _call_rel32(pos + len(b), f_filter)            # call f_filter
    b += bytes.fromhex('84 c0 75 51')                   # test al,al; jne after
    b += _call_rel32(pos + len(b), f_cen)               # call f_cen
    b += bytes.fromhex('84 c0 74 23')                   # test al,al; je sink
    b += _mov_r32_rip(pos + len(b), 0x0d, ch)           # mov ecx, [ch]
    b += _call_rel32(pos + len(b), f_verb)              # call f_verb
    b += bytes.fromhex('85 c0 7f 14')                   # test eax,eax; jg sink
    b += bytes.fromhex('41 b8') + struct.pack('<I', arg)  # mov r8d, arg
    b += _lea_rip(pos + len(b), 0x15, fmt)              # lea rdx, [fmt]
    b += bytes.fromhex('33 c9')                         # xor ecx, ecx
    b += _call_rel32(pos + len(b), f_con)               # call f_con
    b += _mov_r32_rip(pos + len(b), 0x15, ch)           # mov edx, [ch]
    b += bytes.fromhex('33 c9')                         # xor ecx, ecx
    b += _call_rel32(pos + len(b), f_sen)               # call f_sen
    b += bytes.fromhex('84 c0 74 14')                   # test al,al; je after
    b += bytes.fromhex('41 b8') + struct.pack('<I', arg)  # mov r8d, arg
    b += _lea_rip(pos + len(b), 0x15, fmt)              # lea rdx, [fmt]
    b += bytes.fromhex('33 c9')                         # xor ecx, ecx
    b += _call_rel32(pos + len(b), f_sink)              # call f_sink
    return bytes(b)


def mk_sbo_payload_binder(rva, base_vt, der_vt, sbo_cl, heap_cl, ch, f_filter, f_cen, f_verb, f_con, f_sen, f_sink, fmt, alloc_fn, cap=16):
    b = bytearray()
    b += bytes.fromhex('48 89 5c 24 08 57 48 83 ec 20 48 8b fa 48 8b d9 49 83 f9') + bytes([cap]) + bytes.fromhex('72 37')
    b += _lea_rip(rva + len(b), 0x0d, base_vt)
    b += bytes.fromhex('4c 89 03 49 89 08')
    b += _lea_rip(rva + len(b), 0x0d, der_vt)
    b += bytes.fromhex('48 8b 02 49 89 40 08')
    b += _lea_rip(rva + len(b), 0x05, sbo_cl)
    b += bytes.fromhex('48 89 43 08 48 8b c3 49 89 08 48 8b 5c 24 30 48 83 c4 20 5f c3')
    b += _guarded_log_u64(rva + len(b), ch, f_filter, f_cen, f_verb, f_con, f_sen, f_sink, fmt, cap)
    b += bytes([0xb9]) + struct.pack('<I', cap)
    b += _call_rip_ind(rva + len(b), alloc_fn)
    b += bytes.fromhex('48 89 03')
    b += _lea_rip(rva + len(b), 0x0d, base_vt)
    b += bytes.fromhex('48 89 08 48 8b 0f 48 89 48 08')
    b += _lea_rip(rva + len(b), 0x0d, der_vt)
    b += bytes.fromhex('48 89 08')
    b += _lea_rip(rva + len(b), 0x05, heap_cl)
    b += bytes.fromhex('48 89 43 08 48 8b c3 48 8b 5c 24 30 48 83 c4 20 5f c3')
    return bytes(b)


def mk_sbo_empty_binder(rva, vt, sbo_cl, heap_cl, ch, f_filter, f_cen, f_verb, f_con, f_sen, f_sink, fmt, alloc_fn, cap=16):
    b = bytearray()
    b += bytes.fromhex('40 53 48 83 ec 20 48 8b d9 49 83 f9') + bytes([cap]) + bytes.fromhex('72 21')
    b += _lea_rip(rva + len(b), 0x05, sbo_cl)
    b += bytes.fromhex('4c 89 03')
    b += _lea_rip(rva + len(b), 0x0d, vt)
    b += bytes.fromhex('48 89 43 08 48 8b c3 49 89 08 48 83 c4 20 5b c3')
    b += _guarded_log_u64(rva + len(b), ch, f_filter, f_cen, f_verb, f_con, f_sen, f_sink, fmt, cap)
    b += bytes([0xb9]) + struct.pack('<I', cap)
    b += _call_rip_ind(rva + len(b), alloc_fn)
    b += bytes.fromhex('48 89 03')
    b += _lea_rip(rva + len(b), 0x0d, vt)
    b += bytes.fromhex('48 89 08')
    b += _lea_rip(rva + len(b), 0x05, heap_cl)
    b += bytes.fromhex('48 89 43 08 48 8b c3 48 83 c4 20 5b c3')
    return bytes(b)


def mk_sbo_payload_cloner(rva, alloc_fn, base_vt, der_vt, alloc_bytes=16):
    b = bytearray()
    b += bytes.fromhex('40 53 48 83 ec 20 48 8b d9 b9') + struct.pack('<I', alloc_bytes)
    b += _call_rip_ind(rva + len(b), alloc_fn)
    b += _lea_rip(rva + len(b), 0x0d, base_vt)
    b += bytes.fromhex('48 89 08 48 8b 0b 48 89 48 08')
    b += _lea_rip(rva + len(b), 0x0d, der_vt)
    b += bytes.fromhex('48 89 08 48 83 c4 20 5b c3')
    return bytes(b)


def mk_guarded_log_emit(rva, ch, lvl, f_filter, f_cen, f_verb, f_con, f_sen, f_sink, fmt, *, ch2=None):
    b = bytearray()
    b += bytes.fromhex('84 d2 75 76 53 48 83 ec 20')
    b += _mov_r32_rip(rva + len(b), 0x0d, ch)
    b += bytes([0xba]) + struct.pack('<I', lvl)
    b += bytes.fromhex('49 8b d8')
    b += _call_rel32(rva + len(b), f_filter)
    b += bytes.fromhex('84 c0 75 55')
    b += _call_rel32(rva + len(b), f_cen)
    b += bytes.fromhex('84 c0 74 24')
    b += _mov_r32_rip(rva + len(b), 0x0d, ch if ch2 is None else ch2)
    b += _call_rel32(rva + len(b), f_verb)
    b += bytes.fromhex('83 f8') + bytes([lvl]) + bytes.fromhex('7f 14')
    b += bytes.fromhex('4c 8b c3')
    b += _lea_rip(rva + len(b), 0x15, fmt)
    b += bytes([0xb9]) + struct.pack('<I', lvl)
    b += _call_rel32(rva + len(b), f_con)
    b += _mov_r32_rip(rva + len(b), 0x15, ch)
    b += bytes([0xb9]) + struct.pack('<I', lvl)
    b += _call_rel32(rva + len(b), f_sen)
    b += bytes.fromhex('84 c0 74 14')
    b += bytes.fromhex('4c 8b c3')
    b += _lea_rip(rva + len(b), 0x15, fmt)
    b += bytes([0xb9]) + struct.pack('<I', lvl)
    b += _call_rel32(rva + len(b), f_sink)
    b += bytes.fromhex('48 83 c4 20 5b c3')
    return bytes(b)


def mk_null_log_forward(rva, ch, lvl, f_filter, f_cen, f_verb, f_con, f_sen, f_sink, fmt, arg, f_strlen, f_fwd):
    b = bytearray()
    b += bytes.fromhex('48 89 5c 24 08 57 48 83 ec 20 48 8b da 48 8b f9 48 85 d2 75 77')
    b += _mov_r32_rip(rva + len(b), 0x0d, ch)
    b += bytes.fromhex('8d 53') + bytes([lvl])
    b += _call_rel32(rva + len(b), f_filter)
    b += bytes.fromhex('84 c0 75 7b')
    b += _call_rel32(rva + len(b), f_cen)
    b += bytes.fromhex('84 c0 74 26')
    b += _mov_r32_rip(rva + len(b), 0x0d, ch)
    b += _call_rel32(rva + len(b), f_verb)
    b += bytes.fromhex('83 f8') + bytes([lvl]) + bytes.fromhex('7f 16')
    b += bytes([0x4c, 0x8d, 0x05]) + _rel32(rva + len(b), 7, arg)
    b += _lea_rip(rva + len(b), 0x15, fmt)
    b += bytes.fromhex('8d 4b') + bytes([lvl])
    b += _call_rel32(rva + len(b), f_con)
    b += _mov_r32_rip(rva + len(b), 0x15, ch)
    b += bytes([0xb9]) + struct.pack('<I', lvl)
    b += _call_rel32(rva + len(b), f_sen)
    b += bytes.fromhex('84 c0 74 38')
    b += bytes([0x4c, 0x8d, 0x05]) + _rel32(rva + len(b), 7, arg)
    b += bytes([0xb9]) + struct.pack('<I', lvl)
    b += _lea_rip(rva + len(b), 0x15, fmt)
    b += bytes.fromhex('48 8b 5c 24 30 48 83 c4 20 5f')
    b += _jmp_rel32(rva + len(b), f_sink)
    b += bytes.fromhex('48 8b cb')
    b += _call_rel32(rva + len(b), f_strlen)
    b += bytes.fromhex('4c 8b c0 48 8b d3 48 8b cf')
    b += _call_rel32(rva + len(b), f_fwd)
    b += bytes.fromhex('48 8b 5c 24 30 48 83 c4 20 5f c3')
    return bytes(b)


def mk_json_field_extract(rva, f_lookup, f_decode, f_fmt, fmt_missing, fmt_type, fmt_type_log,
                          fmt_other, fmt_err_log, ch, lvl, f_filter, f_cen, f_verb, f_con, f_sen, f_sink):
    b = bytearray()
    b += bytes.fromhex('48 89 5c 24 08 48 89 6c 24 10 48 89 74 24 18 57 48 83 ec 30')
    b += bytes.fromhex('33 c0 49 8b f0 48 8b d9 48 89 44 24 20 48 8b 49 08 4c 8d 44 24 20')
    b += bytes.fromhex('48 89 44 24 28 41 0f b6 e9 48 8b fa')
    b += _call_rel32(rva + len(b), f_lookup)
    b += bytes.fromhex('85 c0 75 14 48 8b d6 48 8d 4c 24 20')
    b += _call_rel32(rva + len(b), f_decode)
    b += bytes.fromhex('b0 01 e9 6d 01 00 00')
    b += bytes.fromhex('83 f8 22 75 1e 80 7c 24 60 00 0f 85 5b 01 00 00 4c 8b 03')
    b += _lea_rip(rva + len(b), 0x15, fmt_missing)
    b += bytes.fromhex('48 8b 4b 18 e9 bc 00 00 00')
    b += bytes.fromhex('40 84 ed 0f 85 3f 01 00 00 4c 8b 03 48 8b 4b 18 83 f8 21 0f 85 9c 00 00 00 4c 8b cf')
    b += _lea_rip(rva + len(b), 0x15, fmt_type)
    b += _call_rel32(rva + len(b), f_fmt)
    b += bytes.fromhex('48 8b 43 10 48 b9 01 00 01 00 01 00 00 00 ba') + struct.pack('<I', lvl) + bytes.fromhex('48 89 08')
    b += _mov_r32_rip(rva + len(b), 0x0d, ch)
    b += _call_rel32(rva + len(b), f_filter)
    b += bytes.fromhex('84 c0 0f 85 f7 00 00 00')
    b += _call_rel32(rva + len(b), f_cen)
    b += bytes.fromhex('84 c0 74 27')
    b += _mov_r32_rip(rva + len(b), 0x0d, ch)
    b += _call_rel32(rva + len(b), f_verb)
    b += bytes.fromhex('83 f8') + bytes([lvl]) + bytes.fromhex('7f 17 4c 8b 03')
    b += _lea_rip(rva + len(b), 0x15, fmt_type_log)
    b += bytes.fromhex('4c 8b cf b9') + struct.pack('<I', lvl)
    b += _call_rel32(rva + len(b), f_con)
    b += _mov_r32_rip(rva + len(b), 0x15, ch)
    b += bytes([0xb9]) + struct.pack('<I', lvl)
    b += _call_rel32(rva + len(b), f_sen)
    b += bytes.fromhex('84 c0 0f 84 af 00 00 00 4c 8b 03')
    b += _lea_rip(rva + len(b), 0x15, fmt_type_log)
    b += bytes.fromhex('4c 8b cf b9') + struct.pack('<I', lvl)
    b += _call_rel32(rva + len(b), f_sink)
    b += bytes.fromhex('e9 93 00 00 00')
    b += _lea_rip(rva + len(b), 0x15, fmt_other)
    b += bytes.fromhex('4c 8b cf')
    b += _call_rel32(rva + len(b), f_fmt)
    b += bytes.fromhex('48 8b 43 10 48 b9 02 00 01 00 01 00 00 00 ba') + struct.pack('<I', lvl) + bytes.fromhex('48 89 08')
    b += _mov_r32_rip(rva + len(b), 0x0d, ch)
    b += _call_rel32(rva + len(b), f_filter)
    b += bytes.fromhex('84 c0 75 5f')
    b += _call_rel32(rva + len(b), f_cen)
    b += bytes.fromhex('84 c0 74 29')
    b += _mov_r32_rip(rva + len(b), 0x0d, ch)
    b += _call_rel32(rva + len(b), f_verb)
    b += bytes.fromhex('83 f8') + bytes([lvl]) + bytes.fromhex('7f 19 4c 8b 43 18')
    b += _lea_rip(rva + len(b), 0x15, fmt_err_log)
    b += bytes.fromhex('49 83 c0 10 b9') + struct.pack('<I', lvl)
    b += _call_rel32(rva + len(b), f_con)
    b += _mov_r32_rip(rva + len(b), 0x15, ch)
    b += bytes([0xb9]) + struct.pack('<I', lvl)
    b += _call_rel32(rva + len(b), f_sen)
    b += bytes.fromhex('84 c0 74 19 4c 8b 43 18')
    b += _lea_rip(rva + len(b), 0x15, fmt_err_log)
    b += bytes.fromhex('49 83 c0 10 b9') + struct.pack('<I', lvl)
    b += _call_rel32(rva + len(b), f_sink)
    b += bytes.fromhex('32 c0 48 8b 5c 24 40 48 8b 6c 24 48 48 8b 74 24 50 48 83 c4 30 5f c3')
    return bytes(b)


def mk_json_callback_extract(rva, f_lookup, f_fmt, fmt_type, fmt_type_log, ch, lvl, f_filter, f_cen, f_verb, f_con, f_sen, f_sink):
    b = bytearray()
    b += bytes.fromhex('48 89 5c 24 10 48 89 6c 24 18 48 89 74 24 20 57 48 83 ec 20')
    b += bytes.fromhex('49 8b e8 48 8b d9 48 8b 49 08 4c 8d 44 24 30 49 8b f1 48 8b fa')
    b += _call_rel32(rva + len(b), f_lookup)
    b += bytes.fromhex('85 c0 75 22 4c 8b 4b 18 48 8b d6 4c 8b 43 10 48 8b 4c 24 30 ff d5 48 8b 43 10 48 83 38 00 0f 94 c0 e9 98 00 00 00')
    b += bytes.fromhex('4c 8b 03')
    b += _lea_rip(rva + len(b), 0x15, fmt_type)
    b += bytes.fromhex('48 8b 4b 18 4c 8b cf')
    b += _call_rel32(rva + len(b), f_fmt)
    b += bytes.fromhex('48 8b 43 10 48 b9 01 00 01 00 01 00 00 00 ba') + struct.pack('<I', lvl) + bytes.fromhex('48 89 08')
    b += _mov_r32_rip(rva + len(b), 0x0d, ch)
    b += _call_rel32(rva + len(b), f_filter)
    b += bytes.fromhex('84 c0 75 5b')
    b += _call_rel32(rva + len(b), f_cen)
    b += bytes.fromhex('84 c0 74 27')
    b += _mov_r32_rip(rva + len(b), 0x0d, ch)
    b += _call_rel32(rva + len(b), f_verb)
    b += bytes.fromhex('83 f8') + bytes([lvl]) + bytes.fromhex('7f 17 4c 8b 03')
    b += _lea_rip(rva + len(b), 0x15, fmt_type_log)
    b += bytes.fromhex('4c 8b cf b9') + struct.pack('<I', lvl)
    b += _call_rel32(rva + len(b), f_con)
    b += _mov_r32_rip(rva + len(b), 0x15, ch)
    b += bytes([0xb9]) + struct.pack('<I', lvl)
    b += _call_rel32(rva + len(b), f_sen)
    b += bytes.fromhex('84 c0 74 17 4c 8b 03')
    b += _lea_rip(rva + len(b), 0x15, fmt_type_log)
    b += bytes.fromhex('4c 8b cf b9') + struct.pack('<I', lvl)
    b += _call_rel32(rva + len(b), f_sink)
    b += bytes.fromhex('32 c0 48 8b 5c 24 38 48 8b 6c 24 40 48 8b 74 24 48 48 83 c4 20 5f c3')
    return bytes(b)


def mk_alloc_shared_4(rva, alloc_fn, alloc_bytes, ctor_fn, *, deref_a2=False):
    b = bytearray()
    b += bytes.fromhex('48 89 5c 24 10 48 89 74 24 18 57 48 83 ec 30 48 8b d9 49 8b f8 b9') + struct.pack('<I', alloc_bytes) + bytes.fromhex('48 8b f2')
    b += _call_rip_ind(rva + len(b), alloc_fn)
    b += bytes.fromhex('4c 8b 4b 08 0f 57 c0 48 8b c8 f3 0f 7f 44 24 20 4d 85 c9 74 09 f0 41 ff 41 08 4c 8b 4b 08 48 8b 03 48 8d 54 24 20')
    if deref_a2:
        b += bytes.fromhex('4c 8b 06 4c 89 4c 24 28')
    else:
        b += bytes.fromhex('4c 89 4c 24 28 4c 8b c6')
    b += bytes.fromhex('4c 8b cf 48 89 44 24 20')
    b += _call_rel32(rva + len(b), ctor_fn)
    b += bytes.fromhex('48 8b 5c 24 48 48 8b 74 24 50 48 83 c4 30 5f c3')
    return bytes(b)


def mk_alloc_shared_5(rva, alloc_fn, alloc_bytes, ctor_fn, *, deref_a3=True):
    b = bytearray()
    b += bytes.fromhex('48 89 5c 24 10 48 89 74 24 18 48 89 7c 24 20 41 56 48 83 ec 40 48 8b d9 49 8b f9 b9') + struct.pack('<I', alloc_bytes) + bytes.fromhex('49 8b f0 4c 8b f2')
    b += _call_rip_ind(rva + len(b), alloc_fn)
    b += bytes.fromhex('4c 8b 53 08 0f 57 c0 48 8b c8 f3 0f 7f 44 24 30 4d 85 d2 74 09 f0 41 ff 42 08 4c 8b 53 08 48 8b 03 48 8d 54 24 30')
    if deref_a3:
        b += bytes.fromhex('4c 8b 0e 4d 8b 06')
    else:
        b += bytes.fromhex('4d 8b 06 4c 8b ce')
    b += bytes.fromhex('48 89 44 24 30 4c 89 54 24 38 48 89 7c 24 20')
    b += _call_rel32(rva + len(b), ctor_fn)
    b += bytes.fromhex('48 8b 5c 24 58 48 8b 74 24 60 48 8b 7c 24 68 48 83 c4 40 41 5e c3')
    return bytes(b)


def mk_alloc_tail_2(rva, alloc_fn, alloc_bytes, ctor_fn, *, deref_a1=False):
    b = bytearray()
    b += bytes.fromhex('48 89 5c 24 08 57 48 83 ec 20 48 8b f9 48 8b da b9') + struct.pack('<I', alloc_bytes)
    b += _call_rip_ind(rva + len(b), alloc_fn)
    b += bytes.fromhex('48 8b 17 4c 8b c3') if deref_a1 else bytes.fromhex('4c 8b c3 48 8b d7')
    b += bytes.fromhex('48 8b c8 48 8b 5c 24 30 48 83 c4 20 5f')
    b += _jmp_rel32(rva + len(b), ctor_fn)
    return bytes(b)


def mk_alloc_deref_u32_tail_3(rva, alloc_fn, alloc_bytes, ctor_fn):
    b = bytearray()
    b += bytes.fromhex('48 89 5c 24 08 48 89 74 24 10 57 48 83 ec 20 48 8b f1 49 8b d8 b9') + struct.pack('<I', alloc_bytes) + bytes.fromhex('48 8b fa')
    b += _call_rip_ind(rva + len(b), alloc_fn)
    b += bytes.fromhex('8b 16 4c 8b cb 48 8b c8 4c 8b c7 48 8b 5c 24 30 48 8b 74 24 38 48 83 c4 20 5f')
    b += _jmp_rel32(rva + len(b), ctor_fn)
    return bytes(b)


def mk_alloc_deref2_5(rva, alloc_fn, alloc_bytes, ctor_fn):
    b = bytearray()
    b += bytes.fromhex('48 89 5c 24 08 48 89 6c 24 10 48 89 74 24 18 57 48 83 ec 30 48 8b e9 49 8b d9 b9') + struct.pack('<I', alloc_bytes) + bytes.fromhex('49 8b f8 48 8b f2')
    b += _call_rip_ind(rva + len(b), alloc_fn)
    b += bytes.fromhex('4c 8b 0f 48 8b d5 4c 8b 06 48 8b c8 48 89 5c 24 20')
    b += _call_rel32(rva + len(b), ctor_fn)
    b += bytes.fromhex('48 8b 5c 24 40 48 8b 6c 24 48 48 8b 74 24 50 48 83 c4 30 5f c3')
    return bytes(b)


def mk_buffer_clear_free(rva, alloc_fn, free_fn):
    b = bytearray()
    b += bytes.fromhex('40 53 48 83 ec 20 48 8b d9 48 8b 09 48 85 c9 74 25')
    b += _cmp_rip_qword_0(rva + len(b), alloc_fn)
    b += bytes.fromhex('74 0e')
    b += _mov_rax_rip(rva + len(b), free_fn)
    b += bytes.fromhex('48 85 c0 74 02 ff d0 33 c0 48 89 03 48 89 43 08 48 89 43 10 48 83 c4 20 5b c3')
    return bytes(b)


def mk_buffer_destroy_clear_free(rva, destroy_fn, alloc_fn, free_fn):
    b = bytearray()
    b += bytes.fromhex('40 53 48 83 ec 20 48 8b d9 48 8b 09 48 85 c9 74 34 48 8b 53 08 4c 8b c3')
    b += _call_rel32(rva + len(b), destroy_fn)
    b += _cmp_rip_qword_0(rva + len(b), alloc_fn)
    b += bytes.fromhex('74 11')
    b += _mov_rax_rip(rva + len(b), free_fn)
    b += bytes.fromhex('48 85 c0 74 05 48 8b 0b ff d0 33 c0 48 89 03 48 89 43 08 48 89 43 10 48 83 c4 20 5b c3')
    return bytes(b)


def mk_buffer_reset_free(rva, alloc_fn, free_fn, stride=8):
    sib_bp = {1: 0x2f, 2: 0x6f, 4: 0xaf, 8: 0xef}[stride]
    sib_si = {1: 0x37, 2: 0x77, 4: 0xb7, 8: 0xf7}[stride]
    b = bytearray()
    b += bytes.fromhex('48 89 5c 24 08 48 89 6c 24 10 48 89 74 24 18 57 48 83 ec 20 48 8b d9 49 8b f1 48 8b 09 49 8b e8 48 8b fa 48 85 c9 74 18')
    b += _cmp_rip_qword_0(rva + len(b), alloc_fn)
    b += bytes.fromhex('74 0e')
    b += _mov_rax_rip(rva + len(b), free_fn)
    b += bytes.fromhex('48 85 c0 74 02 ff d0')
    b += bytes([0x48, 0x8d, 0x04, sib_bp])
    b += bytes.fromhex('48 89 3b 48 8b 6c 24 38 48 89 43 08')
    b += bytes([0x48, 0x8d, 0x04, sib_si])
    b += bytes.fromhex('48 8b 74 24 40 48 89 43 10 48 8b 5c 24 30 48 83 c4 20 5f c3')
    return bytes(b)


class SboAllocLogTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('iced_x86'), 'local generator requires iced-x86==1.21.0')
    def test_discover_all_15_templates_and_rejects_negatives(self):
        # Sections: .rdata at 0x1000, .data at 0x2000, .pdata at 0x3000, .text LAST at 0x4000
        rdata = bytes(0x200)
        data = bytes(0x200)
        text = bytearray(0x2800)
        extents = {}
        pdata = bytearray()

        def place(rva, code, *, add_extent=True):
            if add_extent:
                extents[rva] = len(code)
            off = rva - 0x4000
            text[off:off + len(code)] = code

        # Shared helpers in .text (0x4000..0x41ff)
        f_filter, f_cen, f_verb = 0x4000, 0x4010, 0x4020
        f_con, f_sen, f_sink = 0x4030, 0x4040, 0x4050
        f_strlen, f_fwd, f_lookup = 0x4060, 0x4070, 0x4080
        f_decode, f_fmt, f_ctor = 0x4090, 0x40a0, 0x40b0
        f_destroy, f_heap_cl, f_leaf_cl = 0x40c0, 0x40d0, 0x40e0
        for h in (f_filter, f_cen, f_verb, f_con, f_sen, f_sink, f_strlen, f_fwd,
                  f_lookup, f_decode, f_fmt, f_ctor, f_destroy, f_heap_cl):
            place(h, bytes([0xc3]))
        # Leaf cleanup: 16B aligned in .text WITHOUT extent or .pdata
        place(f_leaf_cl, bytes([0xc3]), add_extent=False)

        # Read-only targets in .rdata (0x1000..)
        vt_base, vt_der, fmt_a, fmt_b = 0x1000, 0x1020, 0x1040, 0x1060
        fmt_c, fmt_d, fmt_e, arg_s = 0x1080, 0x10a0, 0x10c0, 0x10e0
        # Unwind info in .rdata: non-EH at 0x1100, EH (UNW_FLAG_EHANDLER=1 -> byte0=0x09) at 0x1110
        rdata_mut = bytearray(0x200)
        rdata_mut[0x100] = 0x01  # version=1, flags=0
        rdata_mut[0x110] = 0x09  # version=1, flags=1 (EHANDLER)
        rdata = bytes(rdata_mut)

        # Writable globals in .data (0x2000..)
        g_ch, g_alloc, g_free = 0x2000, 0x2010, 0x2018

        # Place all 15 positive templates (+ variants = 19 positive functions)
        pos = []
        cursor = 0x4200

        def add_pos(code, expected_base):
            nonlocal cursor
            rva = cursor
            place(rva, code)
            pdata.extend(struct.pack('<III', rva, rva + len(code), 0x1100))
            pos.append((rva, len(code), expected_base))
            cursor = (cursor + len(code) + 0x1f) & ~0xf

        add_pos(mk_sbo_payload_binder(cursor, vt_base, vt_der, f_leaf_cl, f_heap_cl, g_ch,
                                      f_filter, f_cen, f_verb, f_con, f_sen, f_sink, fmt_a, g_alloc),
                generator.SBO_PAYLOAD_BINDER)
        add_pos(mk_sbo_empty_binder(cursor, vt_der, f_leaf_cl, f_heap_cl, g_ch,
                                    f_filter, f_cen, f_verb, f_con, f_sen, f_sink, fmt_a, g_alloc),
                generator.SBO_EMPTY_BINDER)
        add_pos(mk_sbo_payload_cloner(cursor, g_alloc, vt_base, vt_der, 16), generator.SBO_PAYLOAD_CLONER)
        add_pos(mk_guarded_log_emit(cursor, g_ch, 3, f_filter, f_cen, f_verb, f_con, f_sen, f_sink, fmt_a),
                generator.GUARDED_LOG_EMIT)
        add_pos(mk_null_log_forward(cursor, g_ch, 3, f_filter, f_cen, f_verb, f_con, f_sen, f_sink, fmt_a, arg_s, f_strlen, f_fwd),
                generator.NULL_LOG_FORWARD)
        add_pos(mk_json_field_extract(cursor, f_lookup, f_decode, f_fmt, fmt_a, fmt_b, fmt_c, fmt_d, fmt_e,
                                      g_ch, 1, f_filter, f_cen, f_verb, f_con, f_sen, f_sink),
                generator.JSON_FIELD_EXTRACT)
        add_pos(mk_json_callback_extract(cursor, f_lookup, f_fmt, fmt_b, fmt_c, g_ch, 1,
                                         f_filter, f_cen, f_verb, f_con, f_sen, f_sink),
                generator.JSON_CALLBACK_EXTRACT)
        add_pos(mk_alloc_shared_4(cursor, g_alloc, 0x23040, f_ctor, deref_a2=False), generator.ALLOC_SHARED_4)
        add_pos(mk_alloc_shared_4(cursor, g_alloc, 0x23040, f_ctor, deref_a2=True), generator.ALLOC_SHARED_4)
        add_pos(mk_alloc_shared_5(cursor, g_alloc, 0x23040, f_ctor, deref_a3=True), generator.ALLOC_SHARED_5)
        add_pos(mk_alloc_shared_5(cursor, g_alloc, 0x23040, f_ctor, deref_a3=False), generator.ALLOC_SHARED_5)
        add_pos(mk_alloc_tail_2(cursor, g_alloc, 0x170, f_ctor, deref_a1=False), generator.ALLOC_TAIL_2)
        add_pos(mk_alloc_tail_2(cursor, g_alloc, 0x23040, f_ctor, deref_a1=True), generator.ALLOC_TAIL_2)
        add_pos(mk_alloc_deref_u32_tail_3(cursor, g_alloc, 0x23040, f_ctor), generator.ALLOC_DEREF_U32_TAIL_3)
        add_pos(mk_alloc_deref2_5(cursor, g_alloc, 0x31a8, f_ctor), generator.ALLOC_DEREF2_5)
        add_pos(mk_buffer_clear_free(cursor, g_alloc, g_free), generator.BUFFER_CLEAR_FREE)
        add_pos(mk_buffer_destroy_clear_free(cursor, f_destroy, g_alloc, g_free), generator.BUFFER_DESTROY_CLEAR_FREE)
        add_pos(mk_buffer_reset_free(cursor, g_alloc, g_free, stride=8), generator.BUFFER_RESET_FREE)
        add_pos(mk_buffer_reset_free(cursor, g_alloc, g_free, stride=4), generator.BUFFER_RESET_FREE)

        # Negative cases (all must be rejected!)
        def add_neg(code, *, eh=False):
            nonlocal cursor
            rva = cursor
            place(rva, code)
            if eh:
                pdata.extend(struct.pack('<III', rva, rva + len(code), 0x1110))
            cursor = (cursor + len(code) + 0x1f) & ~0xf
            return rva

        # 1. EH handler present in .pdata
        add_neg(mk_sbo_payload_cloner(cursor, g_alloc, vt_base, vt_der, 16), eh=True)
        # 2. Zero alloc_bytes
        add_neg(mk_sbo_payload_cloner(cursor, g_alloc, vt_base, vt_der, 0))
        # 3. Writable vtable (.data instead of .rdata)
        add_neg(mk_sbo_payload_cloner(cursor, g_alloc, g_ch, vt_der, 16))
        # 4. Read-only alloc_fn (.rdata instead of .data)
        add_neg(mk_sbo_payload_cloner(cursor, vt_base, vt_base, vt_der, 16))
        # 5. Unindexed direct callee (mid-instruction)
        add_neg(mk_alloc_tail_2(cursor, g_alloc, 0x170, f_ctor + 1, deref_a1=False))
        # 6. Non-code direct callee (.data)
        extents[g_ch] = 1
        add_neg(mk_alloc_tail_2(cursor, g_alloc, 0x170, g_ch, deref_a1=False))
        # 7. Unaligned non-indexed sbo_cleanup
        add_neg(mk_sbo_empty_binder(cursor, vt_der, f_leaf_cl + 1, f_heap_cl, g_ch,
                                    f_filter, f_cen, f_verb, f_con, f_sen, f_sink, fmt_a, g_alloc))
        # 8. Mismatched log channel in guarded log block
        add_neg(mk_guarded_log_emit(cursor, g_ch, 3, f_filter, f_cen, f_verb, f_con, f_sen, f_sink, fmt_a, ch2=g_alloc))
        # 9. Overlaps matched inventory entry
        matched_rva = add_neg(mk_buffer_clear_free(cursor, g_alloc, g_free))

        image = PeImage(assemble_pe(
            [('.rdata', 0x1000, rdata, 0x40000040),
             ('.data', 0x2000, data, 0xc0000040),
             ('.pdata', 0x3000, bytes(pdata), 0x40000040),
             ('.text', 0x4000, bytes(text), 0x60000020)],
            (0x3000, len(pdata)),
        ))
        inventory = {matched_rva - 1: dict(size=4, status='matched')}
        found = generator.discover(image, extents, inventory)
        self.assertEqual([(m.rva, m.size, m.base_family) for m in found], pos)

    @unittest.skipUnless(importlib.util.find_spec('iced_x86'), 'local generator requires iced-x86==1.21.0')
    def test_discover_shards_large_families(self):
        count = generator.SHARD_SIZE + 2
        text = bytearray(0x60 * (count + 4))
        extents = {0x4000: 1}
        text[0] = 0xc3
        for idx in range(count):
            rva = 0x4040 + idx * 0x60
            code = mk_sbo_payload_cloner(rva, 0x2010, 0x1000, 0x1020, 16)
            extents[rva] = len(code)
            text[rva - 0x4000:rva - 0x4000 + len(code)] = code
        image = PeImage(assemble_pe(
            [('.rdata', 0x1000, bytes(0x100), 0x40000040),
             ('.data', 0x2000, bytes(0x100), 0xc0000040),
             ('.text', 0x4000, bytes(text), 0x60000020)],
            (0, 0),
        ))
        members = generator.discover(image, extents, {})
        self.assertEqual(len(members), count)
        self.assertEqual([m.family for m in members[:generator.SHARD_SIZE]], [f'{generator.SBO_PAYLOAD_CLONER}1'] * generator.SHARD_SIZE)
        self.assertEqual([m.family for m in members[generator.SHARD_SIZE:]], [f'{generator.SBO_PAYLOAD_CLONER}2'] * 2)

    def test_emit_produces_clean_units_and_bindings(self):
        members = [
            generator.Member(0x4200, 241, generator.SBO_PAYLOAD_BINDER, alloc_fn=0x2010, alloc_bytes=16,
                             base_vtable=0x1000, derived_vtable=0x1020, sbo_cleanup=0x40e0, heap_cleanup=0x40d0,
                             log_channel=0x2000, log_level=0, log_filter=0x4000, log_console_en=0x4010,
                             log_verbosity=0x4020, log_console=0x4030, log_sink_en=0x4040, log_sink=0x4050, log_fmt=0x1040),
            generator.Member(0x4300, 190, generator.SBO_EMPTY_BINDER, alloc_fn=0x2010, alloc_bytes=16,
                             derived_vtable=0x1020, sbo_cleanup=0x40e0, heap_cleanup=0x40d0,
                             log_channel=0x2000, log_level=0, log_filter=0x4000, log_console_en=0x4010,
                             log_verbosity=0x4020, log_console=0x4030, log_sink_en=0x4040, log_sink=0x4050, log_fmt=0x1040),
            generator.Member(0x4400, 53, generator.SBO_PAYLOAD_CLONER, alloc_fn=0x2010, alloc_bytes=16,
                             base_vtable=0x1000, derived_vtable=0x1020),
            generator.Member(0x4480, 123, generator.GUARDED_LOG_EMIT, log_channel=0x2000, log_level=3,
                             log_filter=0x4000, log_console_en=0x4010, log_verbosity=0x4020,
                             log_console=0x4030, log_sink_en=0x4040, log_sink=0x4050, log_fmt=0x1040),
            generator.Member(0x4520, 173, generator.NULL_LOG_FORWARD, log_channel=0x2000, log_level=3,
                             log_filter=0x4000, log_console_en=0x4010, log_verbosity=0x4020,
                             log_console=0x4030, log_sink_en=0x4040, log_sink=0x4050, log_fmt=0x1040,
                             log_arg=0x10e0, callee=0x4060, callee2=0x4070),
            generator.Member(0x4600, 469, generator.JSON_FIELD_EXTRACT, callee=0x4080, callee2=0x4090, callee3=0x40a0,
                             missing_rc=0x22, type_rc=0x21, fmt_missing=0x1040, fmt_type=0x1060, fmt_type_log=0x1080,
                             fmt_other=0x10a0, fmt_err_log=0x10c0, err_code_type=0x1000000010001, err_code_other=0x1000000010002,
                             log_channel=0x2000, log_level=1, log_filter=0x4000, log_console_en=0x4010,
                             log_verbosity=0x4020, log_console=0x4030, log_sink_en=0x4040, log_sink=0x4050),
            generator.Member(0x4800, 257, generator.JSON_CALLBACK_EXTRACT, callee=0x4080, callee3=0x40a0,
                             fmt_type=0x1060, fmt_type_log=0x1080, err_code_type=0x1000000010001,
                             log_channel=0x2000, log_level=1, log_filter=0x4000, log_console_en=0x4010,
                             log_verbosity=0x4020, log_console=0x4030, log_sink_en=0x4040, log_sink=0x4050),
            generator.Member(0x4920, 110, generator.ALLOC_SHARED_4, alloc_fn=0x2010, alloc_bytes=0x23040, callee=0x40b0, deref_a2=False),
            generator.Member(0x49a0, 110, generator.ALLOC_SHARED_4, alloc_fn=0x2010, alloc_bytes=0x23040, callee=0x40b0, deref_a2=True),
            generator.Member(0x4a20, 130, generator.ALLOC_SHARED_5, alloc_fn=0x2010, alloc_bytes=0x23040, callee=0x40b0, deref_a2=True, deref_a3=True),
            generator.Member(0x4ac0, 130, generator.ALLOC_SHARED_5, alloc_fn=0x2010, alloc_bytes=0x23040, callee=0x40b0, deref_a2=True, deref_a3=False),
            generator.Member(0x4b60, 51, generator.ALLOC_TAIL_2, alloc_fn=0x2010, alloc_bytes=0x170, callee=0x40b0, deref_a1=False),
            generator.Member(0x4ba0, 51, generator.ALLOC_TAIL_2, alloc_fn=0x2010, alloc_bytes=0x23040, callee=0x40b0, deref_a1=True),
            generator.Member(0x4be0, 66, generator.ALLOC_DEREF_U32_TAIL_3, alloc_fn=0x2010, alloc_bytes=0x23040, callee=0x40b0),
            generator.Member(0x4c40, 86, generator.ALLOC_DEREF2_5, alloc_fn=0x2010, alloc_bytes=0x31a8, callee=0x40b0),
            generator.Member(0x4ca0, 60, generator.BUFFER_CLEAR_FREE, alloc_fn=0x2010, free_fn=0x2018),
            generator.Member(0x4ce0, 75, generator.BUFFER_DESTROY_CLEAR_FREE, alloc_fn=0x2010, free_fn=0x2018, callee=0x40c0),
            generator.Member(0x4d40, 104, generator.BUFFER_RESET_FREE, alloc_fn=0x2010, free_fn=0x2018, elem_stride=8),
            generator.Member(0x4dc0, 104, generator.BUFFER_RESET_FREE, alloc_fn=0x2010, free_fn=0x2018, elem_stride=4),
        ]
        by_fam = {}
        for m in members:
            by_fam.setdefault(m.family, []).append(m)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for fam, rows in by_fam.items():
                task, directory = generator.emit(root, fam, rows, 'synthetic', 'synthetic')
                src = root / task['compiled']
                hdr = src.with_suffix('.h')
                text = src.read_text()
                symbols = json.loads((directory / 'symbols.json').read_text())['symbols']
                for m in rows:
                    self.assertIn(f'// 0x{m.rva:x} {fam}::{m.name}', text)
                    sym = generator._mangled_symbol(fam, m)
                    self.assertEqual(symbols[sym], dict(rva=m.rva, size=m.size))
                self.assertEqual(contrib_screen.screen_tree(root, task['compiled']), [])
                for path in (src, hdr):
                    self.assertEqual(lint_reconstruction.lint_file(
                        root, path.relative_to(root).as_posix(), 'synthetic', compact=True), [])
            units = check_code_rules.analyse(root)
            match_findings = [(name, f) for name, unit in units.items()
                              for f in unit.findings if f[1] in check_code_rules.MATCH_RULES]
            self.assertEqual(match_findings, [])


if __name__ == '__main__':
    unittest.main()
