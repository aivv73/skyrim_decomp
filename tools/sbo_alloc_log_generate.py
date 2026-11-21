"""Draft BNet-runtime SBO binder, custom-allocator factory, buffer-clear, and guarded-log wrappers.

    python tools/sbo_alloc_log_generate.py --exe /path/to/SkyrimSE.exe
    python tools/sbo_alloc_log_generate.py --exe /path/to/SkyrimSE.exe --out .local/sbo-alloc-log --compare

Requires iced-x86==1.21.0. --compare uses the pinned contributor compiler and
checks the full code, unwind information, static storage references and direct
call targets for every generated member.
"""
from dataclasses import dataclass, replace
from types import SimpleNamespace

import leaf_family_batch
from pecoff import Reject, SCN_MEM_EXECUTE

SCN_MEM_WRITE = 0x80000000

SBO_PAYLOAD_BINDER = 'SboPayloadBinders'
SBO_EMPTY_BINDER = 'SboEmptyBinders'
SBO_PAYLOAD_CLONER = 'SboPayloadCloners'
GUARDED_LOG_EMIT = 'GuardedLogEmitters'
NULL_LOG_FORWARD = 'NullCheckedLogForwarders'
JSON_FIELD_EXTRACT = 'JsonFieldExtractors'
JSON_CALLBACK_EXTRACT = 'JsonCallbackExtractors'
ALLOC_SHARED_4 = 'AllocSharedConstruct4Wrappers'
ALLOC_SHARED_5 = 'AllocSharedConstruct5Wrappers'
ALLOC_TAIL_2 = 'AllocTailConstruct2Wrappers'
ALLOC_DEREF_U32_TAIL_3 = 'AllocDerefU32Tail3Wrappers'
ALLOC_DEREF2_5 = 'AllocDeref2Construct5Wrappers'
BUFFER_CLEAR_FREE = 'BufferClearFreeWrappers'
BUFFER_DESTROY_CLEAR_FREE = 'BufferDestroyClearFreeWrappers'
BUFFER_RESET_FREE = 'BufferResetFreeWrappers'

BASE_FAMILIES = (
    SBO_PAYLOAD_BINDER,
    SBO_EMPTY_BINDER,
    SBO_PAYLOAD_CLONER,
    GUARDED_LOG_EMIT,
    NULL_LOG_FORWARD,
    JSON_FIELD_EXTRACT,
    JSON_CALLBACK_EXTRACT,
    ALLOC_SHARED_4,
    ALLOC_SHARED_5,
    ALLOC_TAIL_2,
    ALLOC_DEREF_U32_TAIL_3,
    ALLOC_DEREF2_5,
    BUFFER_CLEAR_FREE,
    BUFFER_DESTROY_CLEAR_FREE,
    BUFFER_RESET_FREE,
)

SHARD_SIZE = 20
CANDIDATE_SIZES = frozenset((51, 53, 60, 66, 75, 86, 104, 110, 123, 130, 173, 190, 241, 257, 469))


@dataclass(frozen=True)
class Member:
    rva: int
    size: int
    family: str
    # Allocator / buffer bindings
    alloc_fn: int | None = None
    free_fn: int | None = None
    alloc_bytes: int | None = None
    elem_stride: int | None = None
    # SBO / vtable bindings
    base_vtable: int | None = None
    derived_vtable: int | None = None
    sbo_cleanup: int | None = None
    heap_cleanup: int | None = None
    # Guarded logger bindings
    log_channel: int | None = None
    log_level: int | None = None
    log_filter: int | None = None
    log_console_en: int | None = None
    log_verbosity: int | None = None
    log_console: int | None = None
    log_sink_en: int | None = None
    log_sink: int | None = None
    log_fmt: int | None = None
    log_arg: int | None = None
    # Callee / JSON bindings
    callee: int | None = None
    callee2: int | None = None
    callee3: int | None = None
    deref_a1: bool = False
    deref_a2: bool = False
    deref_a3: bool = False
    missing_rc: int | None = None
    type_rc: int | None = None
    fmt_missing: int | None = None
    fmt_type: int | None = None
    fmt_type_log: int | None = None
    fmt_other: int | None = None
    fmt_err_log: int | None = None
    err_code_type: int | None = None
    err_code_other: int | None = None

    @property
    def name(self):
        return f'Wrap{self.rva:x}'

    @property
    def base_family(self):
        return self.family.rstrip('0123456789')


def _rip(ins, op_index=None):
    from iced_x86 import OpKind, Register
    if ins.memory_base != Register.RIP:
        return None
    if op_index is not None and ins.op_kind(op_index) != OpKind.MEMORY:
        return None
    return ins.ip_rel_memory_address


def _call_target(ins):
    from iced_x86 import Code
    return ins.near_branch_target if ins.code == Code.CALL_REL32_64 else None


def _jmp_target(ins):
    from iced_x86 import Code
    return ins.near_branch_target if ins.code == Code.JMP_REL32_64 else None


def _is_mov_reg_reg(ins, dst, src):
    from iced_x86 import Code
    return (ins.code == Code.MOV_R64_RM64 and ins.op0_register == dst and ins.op1_register == src)


def _is_lea_rip(ins, dst):
    from iced_x86 import Code, Register
    if ins.code == Code.LEA_R64_M and ins.op0_register == dst and ins.memory_base == Register.RIP:
        return ins.ip_rel_memory_address
    return None


def _is_mov_r32_rip(ins, dst):
    from iced_x86 import Code, Register
    if ins.code == Code.MOV_R32_RM32 and ins.op0_register == dst and ins.memory_base == Register.RIP:
        return ins.ip_rel_memory_address
    return None


def _is_mov_r64_rip(ins, dst):
    from iced_x86 import Code, Register
    if ins.code == Code.MOV_R64_RM64 and ins.op0_register == dst and ins.memory_base == Register.RIP:
        return ins.ip_rel_memory_address
    return None


def _is_mov_r32_imm(ins, dst):
    from iced_x86 import Code
    if ins.code == Code.MOV_R32_IMM32 and ins.op0_register == dst:
        return ins.immediate32
    return None


def _is_xor_r32_self(ins, reg):
    from iced_x86 import Code
    return ins.code == Code.XOR_R32_RM32 and ins.op0_register == reg and ins.op1_register == reg


def _is_level_set(ins, reg, zero_reg=None):
    from iced_x86 import Code, OpKind, Register
    if _is_xor_r32_self(ins, reg):
        return 0
    if ins.code == Code.MOV_R32_IMM32 and ins.op0_register == reg:
        return ins.immediate32
    if (zero_reg is not None and ins.code == Code.LEA_R32_M and ins.op0_register == reg
            and ins.memory_base == zero_reg and ins.memory_index == Register.NONE):
        return ins.memory_displacement
    return None


def _is_level_cmp_eax(ins):
    from iced_x86 import Code, Register
    if ins.code == Code.TEST_RM32_R32 and ins.op0_register == Register.EAX and ins.op1_register == Register.EAX:
        return 0
    if ins.code == Code.CMP_RM32_IMM8 and ins.op0_register == Register.EAX:
        return ins.immediate8to32
    if ins.code == Code.CMP_EAX_IMM32 and ins.op0_register == Register.EAX:
        return ins.immediate32
    return None


def _is_call_rip_indirect(ins):
    from iced_x86 import Code, OpKind, Register
    if ins.code == Code.CALL_RM64 and ins.op0_kind == OpKind.MEMORY and ins.memory_base == Register.RIP:
        return ins.ip_rel_memory_address
    return None


def _is_mem_r64(ins, code, op_reg_idx, reg, base, disp=0):
    from iced_x86 import OpKind, Register
    if ins.code != code:
        return False
    mem_idx = 1 - op_reg_idx
    return (ins.op_kind(op_reg_idx) == OpKind.REGISTER and ins.op_register(op_reg_idx) == reg
            and ins.op_kind(mem_idx) == OpKind.MEMORY and ins.memory_base == base
            and ins.memory_index == Register.NONE and ins.memory_displacement == disp)


def _is_test_reg_self(ins, code, reg):
    return ins.code == code and ins.op0_register == reg and ins.op1_register == reg


def _is_branch(ins, code, target):
    return ins.code == code and ins.near_branch_target == target


def _is_stack_save(ins, reg, disp):
    from iced_x86 import Code, Register
    return _is_mem_r64(ins, Code.MOV_RM64_R64, 1, reg, Register.RSP, disp)


def _is_stack_load(ins, reg, disp):
    from iced_x86 import Code, Register
    return _is_mem_r64(ins, Code.MOV_R64_RM64, 0, reg, Register.RSP, disp)


def _is_sub_rsp(ins, imm):
    from iced_x86 import Code, Register
    return ins.code == Code.SUB_RM64_IMM8 and ins.op0_register == Register.RSP and ins.immediate8to64 == imm


def _is_add_rsp(ins, imm):
    from iced_x86 import Code, Register
    return ins.code == Code.ADD_RM64_IMM8 and ins.op0_register == Register.RSP and ins.immediate8to64 == imm


def _is_push(ins, reg):
    from iced_x86 import Code
    return ins.code == Code.PUSH_R64 and ins.op0_register == reg


def _is_pop(ins, reg):
    from iced_x86 import Code
    return ins.code == Code.POP_R64 and ins.op0_register == reg


def _match_guarded_log_u64(insns, idx, after_ip):
    """Match 25-instruction guarded console+sink log block with (level=0, fmt, u64_arg).
    Returns (next_idx, log_channel, log_level, log_filter, log_console_en, log_verbosity,
             log_console, log_sink_en, log_sink, fmt_rva, arg_u64) or None."""
    from iced_x86 import Code, Register
    if idx + 25 > len(insns):
        return None
    ch = _is_mov_r32_rip(insns[idx], Register.ECX)
    lvl = _is_level_set(insns[idx + 1], Register.EDX)
    f_filter = _call_target(insns[idx + 2])
    if ch is None or lvl != 0 or f_filter is None:
        return None
    if not _is_test_reg_self(insns[idx + 3], Code.TEST_RM8_R8, Register.AL):
        return None
    if not _is_branch(insns[idx + 4], Code.JNE_REL8_64, after_ip):
        return None
    f_cen = _call_target(insns[idx + 5])
    if f_cen is None or not _is_test_reg_self(insns[idx + 6], Code.TEST_RM8_R8, Register.AL):
        return None
    sink_ip = insns[idx + 16].ip
    if not _is_branch(insns[idx + 7], Code.JE_REL8_64, sink_ip):
        return None
    if _is_mov_r32_rip(insns[idx + 8], Register.ECX) != ch:
        return None
    f_verb = _call_target(insns[idx + 9])
    if f_verb is None or _is_level_cmp_eax(insns[idx + 10]) != lvl:
        return None
    if not _is_branch(insns[idx + 11], Code.JG_REL8_64, sink_ip):
        return None
    arg = _is_mov_r32_imm(insns[idx + 12], Register.R8D)
    fmt = _is_lea_rip(insns[idx + 13], Register.RDX)
    if arg is None or fmt is None or _is_level_set(insns[idx + 14], Register.ECX) != lvl:
        return None
    f_con = _call_target(insns[idx + 15])
    if f_con is None:
        return None
    if _is_mov_r32_rip(insns[idx + 16], Register.EDX) != ch:
        return None
    if _is_level_set(insns[idx + 17], Register.ECX) != lvl:
        return None
    f_sen = _call_target(insns[idx + 18])
    if f_sen is None or not _is_test_reg_self(insns[idx + 19], Code.TEST_RM8_R8, Register.AL):
        return None
    if not _is_branch(insns[idx + 20], Code.JE_REL8_64, after_ip):
        return None
    if _is_mov_r32_imm(insns[idx + 21], Register.R8D) != arg:
        return None
    if _is_lea_rip(insns[idx + 22], Register.RDX) != fmt:
        return None
    if _is_level_set(insns[idx + 23], Register.ECX) != lvl:
        return None
    f_sink = _call_target(insns[idx + 24])
    if f_sink is None:
        return None
    return (idx + 25, ch, lvl, f_filter, f_cen, f_verb, f_con, f_sen, f_sink, fmt, arg)


def _classify_sbo_payload_binder(insns, rva, size):
    """241B SBO vs logged-heap 16B payload wrapper binder (62 instructions)."""
    from iced_x86 import Code, Register
    if size != 241 or len(insns) != 62:
        return None
    if not (_is_stack_save(insns[0], Register.RBX, 8)
            and _is_push(insns[1], Register.RDI)
            and _is_sub_rsp(insns[2], 0x20)
            and _is_mov_reg_reg(insns[3], Register.RDI, Register.RDX)
            and _is_mov_reg_reg(insns[4], Register.RBX, Register.RCX)):
        return None
    if not (insns[5].code == Code.CMP_RM64_IMM8 and insns[5].op0_register == Register.R9):
        return None
    cap = insns[5].immediate8to64
    if cap != 16:
        return None
    if not _is_branch(insns[6], Code.JB_REL8_64, insns[21].ip):
        return None
    base_vt = _is_lea_rip(insns[7], Register.RCX)
    der_vt = _is_lea_rip(insns[10], Register.RCX)
    sbo_cl = _is_lea_rip(insns[13], Register.RAX)
    if base_vt is None or der_vt is None or sbo_cl is None:
        return None
    if not (_is_mem_r64(insns[8], Code.MOV_RM64_R64, 1, Register.R8, Register.RBX, 0)
            and _is_mem_r64(insns[9], Code.MOV_RM64_R64, 1, Register.RCX, Register.R8, 0)
            and _is_mem_r64(insns[11], Code.MOV_R64_RM64, 0, Register.RAX, Register.RDX, 0)
            and _is_mem_r64(insns[12], Code.MOV_RM64_R64, 1, Register.RAX, Register.R8, 8)
            and _is_mem_r64(insns[14], Code.MOV_RM64_R64, 1, Register.RAX, Register.RBX, 8)
            and _is_mov_reg_reg(insns[15], Register.RAX, Register.RBX)
            and _is_mem_r64(insns[16], Code.MOV_RM64_R64, 1, Register.RCX, Register.R8, 0)
            and _is_stack_load(insns[17], Register.RBX, 0x30)
            and _is_add_rsp(insns[18], 0x20)
            and _is_pop(insns[19], Register.RDI)
            and insns[20].code == Code.RETNQ):
        return None
    log = _match_guarded_log_u64(insns, 21, insns[46].ip)
    if log is None:
        return None
    nxt, ch, lvl, f_filter, f_cen, f_verb, f_con, f_sen, f_sink, fmt, arg = log
    if nxt != 46 or arg != cap:
        return None
    alloc_fn = _is_call_rip_indirect(insns[47])
    if _is_mov_r32_imm(insns[46], Register.ECX) != cap or alloc_fn is None:
        return None
    heap_cl = _is_lea_rip(insns[55], Register.RAX)
    if heap_cl is None:
        return None
    if not (_is_mem_r64(insns[48], Code.MOV_RM64_R64, 1, Register.RAX, Register.RBX, 0)
            and _is_lea_rip(insns[49], Register.RCX) == base_vt
            and _is_mem_r64(insns[50], Code.MOV_RM64_R64, 1, Register.RCX, Register.RAX, 0)
            and _is_mem_r64(insns[51], Code.MOV_R64_RM64, 0, Register.RCX, Register.RDI, 0)
            and _is_mem_r64(insns[52], Code.MOV_RM64_R64, 1, Register.RCX, Register.RAX, 8)
            and _is_lea_rip(insns[53], Register.RCX) == der_vt
            and _is_mem_r64(insns[54], Code.MOV_RM64_R64, 1, Register.RCX, Register.RAX, 0)
            and _is_mem_r64(insns[56], Code.MOV_RM64_R64, 1, Register.RAX, Register.RBX, 8)
            and _is_mov_reg_reg(insns[57], Register.RAX, Register.RBX)
            and _is_stack_load(insns[58], Register.RBX, 0x30)
            and _is_add_rsp(insns[59], 0x20)
            and _is_pop(insns[60], Register.RDI)
            and insns[61].code == Code.RETNQ):
        return None
    return Member(
        rva=rva, size=size, family=SBO_PAYLOAD_BINDER,
        alloc_fn=alloc_fn, alloc_bytes=cap,
        base_vtable=base_vt, derived_vtable=der_vt,
        sbo_cleanup=sbo_cl, heap_cleanup=heap_cl,
        log_channel=ch, log_level=lvl, log_filter=f_filter,
        log_console_en=f_cen, log_verbosity=f_verb, log_console=f_con,
        log_sink_en=f_sen, log_sink=f_sink, log_fmt=fmt,
    )


def _classify_sbo_empty_binder(insns, rva, size):
    """190B SBO vs logged-heap payload-free wrapper binder (50 instructions)."""
    from iced_x86 import Code, Register
    if size != 190 or len(insns) != 50:
        return None
    if not (_is_push(insns[0], Register.RBX)
            and _is_sub_rsp(insns[1], 0x20)
            and _is_mov_reg_reg(insns[2], Register.RBX, Register.RCX)):
        return None
    if not (insns[3].code == Code.CMP_RM64_IMM8 and insns[3].op0_register == Register.R9):
        return None
    cap = insns[3].immediate8to64
    if cap != 16:
        return None
    if not _is_branch(insns[4], Code.JB_REL8_64, insns[14].ip):
        return None
    sbo_cl = _is_lea_rip(insns[5], Register.RAX)
    vt = _is_lea_rip(insns[7], Register.RCX)
    if sbo_cl is None or vt is None:
        return None
    if not (_is_mem_r64(insns[6], Code.MOV_RM64_R64, 1, Register.R8, Register.RBX, 0)
            and _is_mem_r64(insns[8], Code.MOV_RM64_R64, 1, Register.RAX, Register.RBX, 8)
            and _is_mov_reg_reg(insns[9], Register.RAX, Register.RBX)
            and _is_mem_r64(insns[10], Code.MOV_RM64_R64, 1, Register.RCX, Register.R8, 0)
            and _is_add_rsp(insns[11], 0x20)
            and _is_pop(insns[12], Register.RBX)
            and insns[13].code == Code.RETNQ):
        return None
    log = _match_guarded_log_u64(insns, 14, insns[39].ip)
    if log is None:
        return None
    nxt, ch, lvl, f_filter, f_cen, f_verb, f_con, f_sen, f_sink, fmt, arg = log
    if nxt != 39 or arg != cap:
        return None
    alloc_fn = _is_call_rip_indirect(insns[40])
    if _is_mov_r32_imm(insns[39], Register.ECX) != cap or alloc_fn is None:
        return None
    heap_cl = _is_lea_rip(insns[44], Register.RAX)
    if heap_cl is None:
        return None
    if not (_is_mem_r64(insns[41], Code.MOV_RM64_R64, 1, Register.RAX, Register.RBX, 0)
            and _is_lea_rip(insns[42], Register.RCX) == vt
            and _is_mem_r64(insns[43], Code.MOV_RM64_R64, 1, Register.RCX, Register.RAX, 0)
            and _is_mem_r64(insns[45], Code.MOV_RM64_R64, 1, Register.RAX, Register.RBX, 8)
            and _is_mov_reg_reg(insns[46], Register.RAX, Register.RBX)
            and _is_add_rsp(insns[47], 0x20)
            and _is_pop(insns[48], Register.RBX)
            and insns[49].code == Code.RETNQ):
        return None
    return Member(
        rva=rva, size=size, family=SBO_EMPTY_BINDER,
        alloc_fn=alloc_fn, alloc_bytes=cap,
        derived_vtable=vt, sbo_cleanup=sbo_cl, heap_cleanup=heap_cl,
        log_channel=ch, log_level=lvl, log_filter=f_filter,
        log_console_en=f_cen, log_verbosity=f_verb, log_console=f_con,
        log_sink_en=f_sen, log_sink=f_sink, log_fmt=fmt,
    )


def _classify_sbo_payload_cloner(insns, rva, size):
    """53B heap payload wrapper cloner (14 instructions)."""
    from iced_x86 import Code, Register
    if size != 53 or len(insns) != 14:
        return None
    if not (_is_push(insns[0], Register.RBX)
            and _is_sub_rsp(insns[1], 0x20)
            and _is_mov_reg_reg(insns[2], Register.RBX, Register.RCX)):
        return None
    alloc_bytes = _is_mov_r32_imm(insns[3], Register.ECX)
    alloc_fn = _is_call_rip_indirect(insns[4])
    base_vt = _is_lea_rip(insns[5], Register.RCX)
    der_vt = _is_lea_rip(insns[9], Register.RCX)
    if alloc_bytes is None or alloc_bytes == 0 or alloc_fn is None or base_vt is None or der_vt is None:
        return None
    if not (_is_mem_r64(insns[6], Code.MOV_RM64_R64, 1, Register.RCX, Register.RAX, 0)
            and _is_mem_r64(insns[7], Code.MOV_R64_RM64, 0, Register.RCX, Register.RBX, 0)
            and _is_mem_r64(insns[8], Code.MOV_RM64_R64, 1, Register.RCX, Register.RAX, 8)
            and _is_mem_r64(insns[10], Code.MOV_RM64_R64, 1, Register.RCX, Register.RAX, 0)
            and _is_add_rsp(insns[11], 0x20)
            and _is_pop(insns[12], Register.RBX)
            and insns[13].code == Code.RETNQ):
        return None
    return Member(
        rva=rva, size=size, family=SBO_PAYLOAD_CLONER,
        alloc_fn=alloc_fn, alloc_bytes=alloc_bytes,
        base_vtable=base_vt, derived_vtable=der_vt,
    )


def _classify_guarded_log_emit(insns, rva, size):
    """123B guarded log emit wrapper: if (!ok) log(level, fmt, msg) (33 instructions)."""
    from iced_x86 import Code, Register
    if size != 123 or len(insns) != 33:
        return None
    ret_ip = insns[32].ip
    epilogue_ip = insns[30].ip
    sink_ip = insns[21].ip
    if not (_is_test_reg_self(insns[0], Code.TEST_RM8_R8, Register.DL)
            and _is_branch(insns[1], Code.JNE_REL8_64, ret_ip)
            and _is_push(insns[2], Register.RBX)
            and _is_sub_rsp(insns[3], 0x20)):
        return None
    ch = _is_mov_r32_rip(insns[4], Register.ECX)
    lvl = _is_mov_r32_imm(insns[5], Register.EDX)
    f_filter = _call_target(insns[7])
    f_cen = _call_target(insns[10])
    f_verb = _call_target(insns[14])
    fmt = _is_lea_rip(insns[18], Register.RDX)
    f_con = _call_target(insns[20])
    f_sen = _call_target(insns[23])
    f_sink = _call_target(insns[29])
    if None in (ch, lvl, f_filter, f_cen, f_verb, fmt, f_con, f_sen, f_sink) or lvl <= 0:
        return None
    if not (_is_mov_reg_reg(insns[6], Register.RBX, Register.R8)
            and _is_test_reg_self(insns[8], Code.TEST_RM8_R8, Register.AL)
            and _is_branch(insns[9], Code.JNE_REL8_64, epilogue_ip)
            and _is_test_reg_self(insns[11], Code.TEST_RM8_R8, Register.AL)
            and _is_branch(insns[12], Code.JE_REL8_64, sink_ip)
            and _is_mov_r32_rip(insns[13], Register.ECX) == ch
            and _is_level_cmp_eax(insns[15]) == lvl
            and _is_branch(insns[16], Code.JG_REL8_64, sink_ip)
            and _is_mov_reg_reg(insns[17], Register.R8, Register.RBX)
            and _is_mov_r32_imm(insns[19], Register.ECX) == lvl
            and _is_mov_r32_rip(insns[21], Register.EDX) == ch
            and _is_mov_r32_imm(insns[22], Register.ECX) == lvl
            and _is_test_reg_self(insns[24], Code.TEST_RM8_R8, Register.AL)
            and _is_branch(insns[25], Code.JE_REL8_64, epilogue_ip)
            and _is_mov_reg_reg(insns[26], Register.R8, Register.RBX)
            and _is_lea_rip(insns[27], Register.RDX) == fmt
            and _is_mov_r32_imm(insns[28], Register.ECX) == lvl
            and _is_add_rsp(insns[30], 0x20)
            and _is_pop(insns[31], Register.RBX)
            and insns[32].code == Code.RETNQ):
        return None
    return Member(
        rva=rva, size=size, family=GUARDED_LOG_EMIT,
        log_channel=ch, log_level=lvl, log_filter=f_filter,
        log_console_en=f_cen, log_verbosity=f_verb, log_console=f_con,
        log_sink_en=f_sen, log_sink=f_sink, log_fmt=fmt,
    )


def _classify_null_log_forward(insns, rva, size):
    """173B null-checked string forwarder with guarded log tail-call (45 instructions)."""
    from iced_x86 import Code, Register
    if size != 173 or len(insns) != 45:
        return None
    fwd_ip = insns[35].ip
    epi_ip = insns[41].ip
    sink_ip = insns[23].ip
    if not (_is_stack_save(insns[0], Register.RBX, 8)
            and _is_push(insns[1], Register.RDI)
            and _is_sub_rsp(insns[2], 0x20)
            and _is_mov_reg_reg(insns[3], Register.RBX, Register.RDX)
            and _is_mov_reg_reg(insns[4], Register.RDI, Register.RCX)
            and _is_test_reg_self(insns[5], Code.TEST_RM64_R64, Register.RDX)
            and _is_branch(insns[6], Code.JNE_REL8_64, fwd_ip)):
        return None
    ch = _is_mov_r32_rip(insns[7], Register.ECX)
    lvl = _is_level_set(insns[8], Register.EDX, zero_reg=Register.RBX)
    f_filter = _call_target(insns[9])
    f_cen = _call_target(insns[12])
    f_verb = _call_target(insns[16])
    arg = _is_lea_rip(insns[19], Register.R8)
    fmt = _is_lea_rip(insns[20], Register.RDX)
    f_con = _call_target(insns[22])
    f_sen = _call_target(insns[25])
    f_sink = _jmp_target(insns[34])
    f_strlen = _call_target(insns[36])
    f_fwd = _call_target(insns[40])
    if None in (ch, lvl, f_filter, f_cen, f_verb, arg, fmt, f_con, f_sen, f_sink, f_strlen, f_fwd) or lvl <= 0:
        return None
    if not (_is_test_reg_self(insns[10], Code.TEST_RM8_R8, Register.AL)
            and _is_branch(insns[11], Code.JNE_REL8_64, epi_ip)
            and _is_test_reg_self(insns[13], Code.TEST_RM8_R8, Register.AL)
            and _is_branch(insns[14], Code.JE_REL8_64, sink_ip)
            and _is_mov_r32_rip(insns[15], Register.ECX) == ch
            and _is_level_cmp_eax(insns[17]) == lvl
            and _is_branch(insns[18], Code.JG_REL8_64, sink_ip)
            and _is_level_set(insns[21], Register.ECX, zero_reg=Register.RBX) == lvl
            and _is_mov_r32_rip(insns[23], Register.EDX) == ch
            and _is_mov_r32_imm(insns[24], Register.ECX) == lvl
            and _is_test_reg_self(insns[26], Code.TEST_RM8_R8, Register.AL)
            and _is_branch(insns[27], Code.JE_REL8_64, epi_ip)
            and _is_lea_rip(insns[28], Register.R8) == arg
            and _is_mov_r32_imm(insns[29], Register.ECX) == lvl
            and _is_lea_rip(insns[30], Register.RDX) == fmt
            and _is_stack_load(insns[31], Register.RBX, 0x30)
            and _is_add_rsp(insns[32], 0x20)
            and _is_pop(insns[33], Register.RDI)
            and _is_mov_reg_reg(insns[35], Register.RCX, Register.RBX)
            and _is_mov_reg_reg(insns[37], Register.R8, Register.RAX)
            and _is_mov_reg_reg(insns[38], Register.RDX, Register.RBX)
            and _is_mov_reg_reg(insns[39], Register.RCX, Register.RDI)
            and _is_stack_load(insns[41], Register.RBX, 0x30)
            and _is_add_rsp(insns[42], 0x20)
            and _is_pop(insns[43], Register.RDI)
            and insns[44].code == Code.RETNQ):
        return None
    return Member(
        rva=rva, size=size, family=NULL_LOG_FORWARD,
        log_channel=ch, log_level=lvl, log_filter=f_filter,
        log_console_en=f_cen, log_verbosity=f_verb, log_console=f_con,
        log_sink_en=f_sen, log_sink=f_sink, log_fmt=fmt, log_arg=arg,
        callee=f_strlen, callee2=f_fwd,
    )


def _classify_json_field_extract(insns, rva, size):
    """469B JSON field lookup + decode + multi-branch error format & guarded log (110 instructions)."""
    from iced_x86 import Code, OpKind, Register
    if size != 469 or len(insns) != 110:
        return None
    ret_false_ip = insns[103].ip
    epi_ip = insns[104].ip
    if not (_is_stack_save(insns[0], Register.RBX, 8)
            and _is_stack_save(insns[1], Register.RBP, 0x10)
            and _is_stack_save(insns[2], Register.RSI, 0x18)
            and _is_push(insns[3], Register.RDI)
            and _is_sub_rsp(insns[4], 0x30)
            and _is_xor_r32_self(insns[5], Register.EAX)
            and _is_mov_reg_reg(insns[6], Register.RSI, Register.R8)
            and _is_mov_reg_reg(insns[7], Register.RBX, Register.RCX)
            and _is_mem_r64(insns[8], Code.MOV_RM64_R64, 1, Register.RAX, Register.RSP, 0x20)
            and _is_mem_r64(insns[9], Code.MOV_R64_RM64, 0, Register.RCX, Register.RCX, 8)
            and _is_mem_r64(insns[10], Code.LEA_R64_M, 0, Register.R8, Register.RSP, 0x20)
            and _is_mem_r64(insns[11], Code.MOV_RM64_R64, 1, Register.RAX, Register.RSP, 0x28)
            and insns[12].code == Code.MOVZX_R32_RM8 and insns[12].op0_register == Register.EBP
            and insns[12].op1_register == Register.R9L
            and _is_mov_reg_reg(insns[13], Register.RDI, Register.RDX)):
        return None
    f_lookup = _call_target(insns[14])
    f_decode = _call_target(insns[19])
    if f_lookup is None or f_decode is None:
        return None
    if not (_is_test_reg_self(insns[15], Code.TEST_RM32_R32, Register.EAX)
            and _is_branch(insns[16], Code.JNE_REL8_64, insns[22].ip)
            and _is_mov_reg_reg(insns[17], Register.RDX, Register.RSI)
            and _is_mem_r64(insns[18], Code.LEA_R64_M, 0, Register.RCX, Register.RSP, 0x20)
            and insns[20].code == Code.MOV_R8_IMM8 and insns[20].op0_register == Register.AL and insns[20].immediate8 == 1
            and _is_branch(insns[21], Code.JMP_REL32_64, epi_ip)):
        return None
    if not (insns[22].code == Code.CMP_RM32_IMM8 and insns[22].op0_register == Register.EAX):
        return None
    missing_rc = insns[22].immediate8to32
    fmt_missing = _is_lea_rip(insns[27], Register.RDX)
    if not (_is_branch(insns[23], Code.JNE_REL8_64, insns[30].ip)
            and insns[24].code == Code.CMP_RM8_IMM8 and insns[24].op0_kind == OpKind.MEMORY
            and insns[24].memory_base == Register.RSP and insns[24].memory_displacement == 0x60
            and insns[24].immediate8 == 0
            and _is_branch(insns[25], Code.JNE_REL32_64, ret_false_ip)
            and _is_mem_r64(insns[26], Code.MOV_R64_RM64, 0, Register.R8, Register.RBX, 0)
            and fmt_missing is not None
            and _is_mem_r64(insns[28], Code.MOV_R64_RM64, 0, Register.RCX, Register.RBX, 0x18)
            and _is_branch(insns[29], Code.JMP_REL32_64, insns[71].ip)):
        return None
    if not (_is_test_reg_self(insns[30], Code.TEST_RM8_R8, Register.BPL)
            and _is_branch(insns[31], Code.JNE_REL32_64, ret_false_ip)
            and _is_mem_r64(insns[32], Code.MOV_R64_RM64, 0, Register.R8, Register.RBX, 0)
            and _is_mem_r64(insns[33], Code.MOV_R64_RM64, 0, Register.RCX, Register.RBX, 0x18)
            and insns[34].code == Code.CMP_RM32_IMM8 and insns[34].op0_register == Register.EAX):
        return None
    type_rc = insns[34].immediate8to32
    fmt_type = _is_lea_rip(insns[37], Register.RDX)
    f_fmt = _call_target(insns[38])
    if not (_is_branch(insns[35], Code.JNE_REL32_64, insns[70].ip)
            and _is_mov_reg_reg(insns[36], Register.R9, Register.RDI)
            and fmt_type is not None and f_fmt is not None
            and _is_mem_r64(insns[39], Code.MOV_R64_RM64, 0, Register.RAX, Register.RBX, 0x10)
            and insns[40].code == Code.MOV_R64_IMM64 and insns[40].op0_register == Register.RCX):
        return None
    err_code_type = insns[40].immediate64
    lvl = _is_mov_r32_imm(insns[41], Register.EDX)
    ch = _is_mov_r32_rip(insns[43], Register.ECX)
    f_filter = _call_target(insns[44])
    f_cen = _call_target(insns[47])
    f_verb = _call_target(insns[51])
    fmt_type_log = _is_lea_rip(insns[55], Register.RDX)
    f_con = _call_target(insns[58])
    f_sen = _call_target(insns[61])
    f_sink = _call_target(insns[68])
    if None in (lvl, ch, f_filter, f_cen, f_verb, fmt_type_log, f_con, f_sen, f_sink) or lvl <= 0:
        return None
    if not (_is_mem_r64(insns[42], Code.MOV_RM64_R64, 1, Register.RCX, Register.RAX, 0)
            and _is_test_reg_self(insns[45], Code.TEST_RM8_R8, Register.AL)
            and _is_branch(insns[46], Code.JNE_REL32_64, ret_false_ip)
            and _is_test_reg_self(insns[48], Code.TEST_RM8_R8, Register.AL)
            and _is_branch(insns[49], Code.JE_REL8_64, insns[59].ip)
            and _is_mov_r32_rip(insns[50], Register.ECX) == ch
            and _is_level_cmp_eax(insns[52]) == lvl
            and _is_branch(insns[53], Code.JG_REL8_64, insns[59].ip)
            and _is_mem_r64(insns[54], Code.MOV_R64_RM64, 0, Register.R8, Register.RBX, 0)
            and _is_mov_reg_reg(insns[56], Register.R9, Register.RDI)
            and _is_mov_r32_imm(insns[57], Register.ECX) == lvl
            and _is_mov_r32_rip(insns[59], Register.EDX) == ch
            and _is_mov_r32_imm(insns[60], Register.ECX) == lvl
            and _is_test_reg_self(insns[62], Code.TEST_RM8_R8, Register.AL)
            and _is_branch(insns[63], Code.JE_REL32_64, ret_false_ip)
            and _is_mem_r64(insns[64], Code.MOV_R64_RM64, 0, Register.R8, Register.RBX, 0)
            and _is_lea_rip(insns[65], Register.RDX) == fmt_type_log
            and _is_mov_reg_reg(insns[66], Register.R9, Register.RDI)
            and _is_mov_r32_imm(insns[67], Register.ECX) == lvl
            and _is_branch(insns[69], Code.JMP_REL32_64, ret_false_ip)):
        return None
    fmt_other = _is_lea_rip(insns[70], Register.RDX)
    if (fmt_other is None
            or not _is_mov_reg_reg(insns[71], Register.R9, Register.RDI)
            or _call_target(insns[72]) != f_fmt
            or not _is_mem_r64(insns[73], Code.MOV_R64_RM64, 0, Register.RAX, Register.RBX, 0x10)
            or not (insns[74].code == Code.MOV_R64_IMM64 and insns[74].op0_register == Register.RCX)):
        return None
    err_code_other = insns[74].immediate64
    fmt_err_log = _is_lea_rip(insns[89], Register.RDX)
    if fmt_err_log is None:
        return None
    if not (_is_mov_r32_imm(insns[75], Register.EDX) == lvl
            and _is_mem_r64(insns[76], Code.MOV_RM64_R64, 1, Register.RCX, Register.RAX, 0)
            and _is_mov_r32_rip(insns[77], Register.ECX) == ch
            and _call_target(insns[78]) == f_filter
            and _is_test_reg_self(insns[79], Code.TEST_RM8_R8, Register.AL)
            and _is_branch(insns[80], Code.JNE_REL8_64, ret_false_ip)
            and _call_target(insns[81]) == f_cen
            and _is_test_reg_self(insns[82], Code.TEST_RM8_R8, Register.AL)
            and _is_branch(insns[83], Code.JE_REL8_64, insns[93].ip)
            and _is_mov_r32_rip(insns[84], Register.ECX) == ch
            and _call_target(insns[85]) == f_verb
            and _is_level_cmp_eax(insns[86]) == lvl
            and _is_branch(insns[87], Code.JG_REL8_64, insns[93].ip)
            and _is_mem_r64(insns[88], Code.MOV_R64_RM64, 0, Register.R8, Register.RBX, 0x18)
            and insns[90].code == Code.ADD_RM64_IMM8 and insns[90].op0_register == Register.R8 and insns[90].immediate8to64 == 0x10
            and _is_mov_r32_imm(insns[91], Register.ECX) == lvl
            and _call_target(insns[92]) == f_con
            and _is_mov_r32_rip(insns[93], Register.EDX) == ch
            and _is_mov_r32_imm(insns[94], Register.ECX) == lvl
            and _call_target(insns[95]) == f_sen
            and _is_test_reg_self(insns[96], Code.TEST_RM8_R8, Register.AL)
            and _is_branch(insns[97], Code.JE_REL8_64, ret_false_ip)
            and _is_mem_r64(insns[98], Code.MOV_R64_RM64, 0, Register.R8, Register.RBX, 0x18)
            and _is_lea_rip(insns[99], Register.RDX) == fmt_err_log
            and insns[100].code == Code.ADD_RM64_IMM8 and insns[100].op0_register == Register.R8 and insns[100].immediate8to64 == 0x10
            and _is_mov_r32_imm(insns[101], Register.ECX) == lvl
            and _call_target(insns[102]) == f_sink
            and insns[103].code in (Code.XOR_RM8_R8, Code.XOR_R8_RM8) and insns[103].op0_register == Register.AL and insns[103].op1_register == Register.AL
            and _is_stack_load(insns[104], Register.RBX, 0x40)
            and _is_stack_load(insns[105], Register.RBP, 0x48)
            and _is_stack_load(insns[106], Register.RSI, 0x50)
            and _is_add_rsp(insns[107], 0x30)
            and _is_pop(insns[108], Register.RDI)
            and insns[109].code == Code.RETNQ):
        return None
    return Member(
        rva=rva, size=size, family=JSON_FIELD_EXTRACT,
        callee=f_lookup, callee2=f_decode, callee3=f_fmt,
        missing_rc=missing_rc, type_rc=type_rc,
        fmt_missing=fmt_missing, fmt_type=fmt_type, fmt_type_log=fmt_type_log,
        fmt_other=fmt_other, fmt_err_log=fmt_err_log,
        err_code_type=err_code_type, err_code_other=err_code_other,
        log_channel=ch, log_level=lvl, log_filter=f_filter,
        log_console_en=f_cen, log_verbosity=f_verb, log_console=f_con,
        log_sink_en=f_sen, log_sink=f_sink,
    )


def _classify_json_callback_extract(insns, rva, size):
    """257B JSON lookup + callback + error-format & guarded log (65 instructions)."""
    from iced_x86 import Code, OpKind, Register
    if size != 257 or len(insns) != 65:
        return None
    ret_false_ip = insns[58].ip
    epi_ip = insns[59].ip
    if not (_is_stack_save(insns[0], Register.RBX, 0x10)
            and _is_stack_save(insns[1], Register.RBP, 0x18)
            and _is_stack_save(insns[2], Register.RSI, 0x20)
            and _is_push(insns[3], Register.RDI)
            and _is_sub_rsp(insns[4], 0x20)
            and _is_mov_reg_reg(insns[5], Register.RBP, Register.R8)
            and _is_mov_reg_reg(insns[6], Register.RBX, Register.RCX)
            and _is_mem_r64(insns[7], Code.MOV_R64_RM64, 0, Register.RCX, Register.RCX, 8)
            and _is_mem_r64(insns[8], Code.LEA_R64_M, 0, Register.R8, Register.RSP, 0x30)
            and _is_mov_reg_reg(insns[9], Register.RSI, Register.R9)
            and _is_mov_reg_reg(insns[10], Register.RDI, Register.RDX)):
        return None
    f_lookup = _call_target(insns[11])
    if f_lookup is None:
        return None
    if not (_is_test_reg_self(insns[12], Code.TEST_RM32_R32, Register.EAX)
            and _is_branch(insns[13], Code.JNE_REL8_64, insns[23].ip)
            and _is_mem_r64(insns[14], Code.MOV_R64_RM64, 0, Register.R9, Register.RBX, 0x18)
            and _is_mov_reg_reg(insns[15], Register.RDX, Register.RSI)
            and _is_mem_r64(insns[16], Code.MOV_R64_RM64, 0, Register.R8, Register.RBX, 0x10)
            and _is_stack_load(insns[17], Register.RCX, 0x30)
            and insns[18].code == Code.CALL_RM64 and insns[18].op0_kind == OpKind.REGISTER and insns[18].op0_register == Register.RBP
            and _is_mem_r64(insns[19], Code.MOV_R64_RM64, 0, Register.RAX, Register.RBX, 0x10)
            and insns[20].code == Code.CMP_RM64_IMM8 and insns[20].op0_kind == OpKind.MEMORY
            and insns[20].memory_base == Register.RAX and insns[20].memory_displacement == 0 and insns[20].immediate8to64 == 0
            and insns[21].code == Code.SETE_RM8 and insns[21].op0_register == Register.AL
            and _is_branch(insns[22], Code.JMP_REL32_64, epi_ip)):
        return None
    fmt_type = _is_lea_rip(insns[24], Register.RDX)
    f_fmt = _call_target(insns[27])
    if fmt_type is None or f_fmt is None:
        return None
    if not (_is_mem_r64(insns[23], Code.MOV_R64_RM64, 0, Register.R8, Register.RBX, 0)
            and _is_mem_r64(insns[25], Code.MOV_R64_RM64, 0, Register.RCX, Register.RBX, 0x18)
            and _is_mov_reg_reg(insns[26], Register.R9, Register.RDI)
            and _is_mem_r64(insns[28], Code.MOV_R64_RM64, 0, Register.RAX, Register.RBX, 0x10)
            and insns[29].code == Code.MOV_R64_IMM64 and insns[29].op0_register == Register.RCX):
        return None
    err_code_type = insns[29].immediate64
    lvl = _is_mov_r32_imm(insns[30], Register.EDX)
    ch = _is_mov_r32_rip(insns[32], Register.ECX)
    f_filter = _call_target(insns[33])
    f_cen = _call_target(insns[36])
    f_verb = _call_target(insns[40])
    fmt_type_log = _is_lea_rip(insns[44], Register.RDX)
    f_con = _call_target(insns[47])
    f_sen = _call_target(insns[50])
    f_sink = _call_target(insns[57])
    if None in (lvl, ch, f_filter, f_cen, f_verb, fmt_type_log, f_con, f_sen, f_sink) or lvl <= 0:
        return None
    if not (_is_mem_r64(insns[31], Code.MOV_RM64_R64, 1, Register.RCX, Register.RAX, 0)
            and _is_test_reg_self(insns[34], Code.TEST_RM8_R8, Register.AL)
            and _is_branch(insns[35], Code.JNE_REL8_64, ret_false_ip)
            and _is_test_reg_self(insns[37], Code.TEST_RM8_R8, Register.AL)
            and _is_branch(insns[38], Code.JE_REL8_64, insns[48].ip)
            and _is_mov_r32_rip(insns[39], Register.ECX) == ch
            and _is_level_cmp_eax(insns[41]) == lvl
            and _is_branch(insns[42], Code.JG_REL8_64, insns[48].ip)
            and _is_mem_r64(insns[43], Code.MOV_R64_RM64, 0, Register.R8, Register.RBX, 0)
            and _is_mov_reg_reg(insns[45], Register.R9, Register.RDI)
            and _is_mov_r32_imm(insns[46], Register.ECX) == lvl
            and _is_mov_r32_rip(insns[48], Register.EDX) == ch
            and _is_mov_r32_imm(insns[49], Register.ECX) == lvl
            and _is_test_reg_self(insns[51], Code.TEST_RM8_R8, Register.AL)
            and _is_branch(insns[52], Code.JE_REL8_64, ret_false_ip)
            and _is_mem_r64(insns[53], Code.MOV_R64_RM64, 0, Register.R8, Register.RBX, 0)
            and _is_lea_rip(insns[54], Register.RDX) == fmt_type_log
            and _is_mov_reg_reg(insns[55], Register.R9, Register.RDI)
            and _is_mov_r32_imm(insns[56], Register.ECX) == lvl
            and insns[58].code in (Code.XOR_RM8_R8, Code.XOR_R8_RM8) and insns[58].op0_register == Register.AL and insns[58].op1_register == Register.AL
            and _is_stack_load(insns[59], Register.RBX, 0x38)
            and _is_stack_load(insns[60], Register.RBP, 0x40)
            and _is_stack_load(insns[61], Register.RSI, 0x48)
            and _is_add_rsp(insns[62], 0x20)
            and _is_pop(insns[63], Register.RDI)
            and insns[64].code == Code.RETNQ):
        return None
    return Member(
        rva=rva, size=size, family=JSON_CALLBACK_EXTRACT,
        callee=f_lookup, callee3=f_fmt,
        fmt_type=fmt_type, fmt_type_log=fmt_type_log,
        err_code_type=err_code_type,
        log_channel=ch, log_level=lvl, log_filter=f_filter,
        log_console_en=f_cen, log_verbosity=f_verb, log_console=f_con,
        log_sink_en=f_sen, log_sink=f_sink,
    )


def _classify_alloc_shared_4(insns, rva, size):
    """110B custom-alloc + shared_ptr copy + 4-arg construct (direct or deref a2) (29 instructions)."""
    from iced_x86 import Code, OpKind, Register
    if size != 110 or len(insns) != 29:
        return None
    if not (_is_stack_save(insns[0], Register.RBX, 0x10)
            and _is_stack_save(insns[1], Register.RSI, 0x18)
            and _is_push(insns[2], Register.RDI)
            and _is_sub_rsp(insns[3], 0x30)
            and _is_mov_reg_reg(insns[4], Register.RBX, Register.RCX)
            and _is_mov_reg_reg(insns[5], Register.RDI, Register.R8)):
        return None
    alloc_bytes = _is_mov_r32_imm(insns[6], Register.ECX)
    alloc_fn = _is_call_rip_indirect(insns[8])
    ctor_fn = _call_target(insns[23])
    if alloc_bytes is None or alloc_bytes == 0 or alloc_fn is None or ctor_fn is None:
        return None
    if not (_is_mov_reg_reg(insns[7], Register.RSI, Register.RDX)
            and _is_mem_r64(insns[9], Code.MOV_R64_RM64, 0, Register.R9, Register.RBX, 8)
            and insns[10].code == Code.XORPS_XMM_XMMM128 and insns[10].op0_register == Register.XMM0 and insns[10].op1_register == Register.XMM0
            and _is_mov_reg_reg(insns[11], Register.RCX, Register.RAX)
            and insns[12].code == Code.MOVDQU_XMMM128_XMM and insns[12].op1_register == Register.XMM0
            and insns[12].memory_base == Register.RSP and insns[12].memory_displacement == 0x20
            and _is_test_reg_self(insns[13], Code.TEST_RM64_R64, Register.R9)
            and _is_branch(insns[14], Code.JE_REL8_64, insns[17].ip)
            and insns[15].code == Code.INC_RM32 and insns[15].has_lock_prefix
            and insns[15].op0_kind == OpKind.MEMORY and insns[15].memory_base == Register.R9 and insns[15].memory_displacement == 8
            and _is_mem_r64(insns[16], Code.MOV_R64_RM64, 0, Register.R9, Register.RBX, 8)
            and _is_mem_r64(insns[17], Code.MOV_R64_RM64, 0, Register.RAX, Register.RBX, 0)
            and _is_mem_r64(insns[18], Code.LEA_R64_M, 0, Register.RDX, Register.RSP, 0x20)):
        return None
    # Two shapes at insns[19..22]:
    # direct a2: [19] mov [rsp+28h],r9  [20] mov r8,rsi     [21] mov r9,rdi  [22] mov [rsp+20h],rax
    # deref a2:  [19] mov r8,[rsi]      [20] mov [rsp+28h],r9 [21] mov r9,rdi [22] mov [rsp+20h],rax
    if (_is_stack_save(insns[19], Register.R9, 0x28)
            and _is_mov_reg_reg(insns[20], Register.R8, Register.RSI)):
        deref_a2 = False
    elif (_is_mem_r64(insns[19], Code.MOV_R64_RM64, 0, Register.R8, Register.RSI, 0)
            and _is_stack_save(insns[20], Register.R9, 0x28)):
        deref_a2 = True
    else:
        return None
    if not (_is_mov_reg_reg(insns[21], Register.R9, Register.RDI)
            and _is_stack_save(insns[22], Register.RAX, 0x20)
            and _is_stack_load(insns[24], Register.RBX, 0x48)
            and _is_stack_load(insns[25], Register.RSI, 0x50)
            and _is_add_rsp(insns[26], 0x30)
            and _is_pop(insns[27], Register.RDI)
            and insns[28].code == Code.RETNQ):
        return None
    return Member(
        rva=rva, size=size, family=ALLOC_SHARED_4,
        alloc_fn=alloc_fn, alloc_bytes=alloc_bytes,
        callee=ctor_fn, deref_a2=deref_a2,
    )


def _classify_alloc_shared_5(insns, rva, size):
    """130B custom-alloc + shared_ptr copy + 5-arg construct (33 instructions)."""
    from iced_x86 import Code, OpKind, Register
    if size != 130 or len(insns) != 33:
        return None
    if not (_is_stack_save(insns[0], Register.RBX, 0x10)
            and _is_stack_save(insns[1], Register.RSI, 0x18)
            and _is_stack_save(insns[2], Register.RDI, 0x20)
            and _is_push(insns[3], Register.R14)
            and _is_sub_rsp(insns[4], 0x40)
            and _is_mov_reg_reg(insns[5], Register.RBX, Register.RCX)
            and _is_mov_reg_reg(insns[6], Register.RDI, Register.R9)):
        return None
    alloc_bytes = _is_mov_r32_imm(insns[7], Register.ECX)
    alloc_fn = _is_call_rip_indirect(insns[10])
    ctor_fn = _call_target(insns[26])
    if alloc_bytes is None or alloc_bytes == 0 or alloc_fn is None or ctor_fn is None:
        return None
    if not (_is_mov_reg_reg(insns[8], Register.RSI, Register.R8)
            and _is_mov_reg_reg(insns[9], Register.R14, Register.RDX)
            and _is_mem_r64(insns[11], Code.MOV_R64_RM64, 0, Register.R10, Register.RBX, 8)
            and insns[12].code == Code.XORPS_XMM_XMMM128 and insns[12].op0_register == Register.XMM0 and insns[12].op1_register == Register.XMM0
            and _is_mov_reg_reg(insns[13], Register.RCX, Register.RAX)
            and insns[14].code == Code.MOVDQU_XMMM128_XMM and insns[14].op1_register == Register.XMM0
            and insns[14].memory_base == Register.RSP and insns[14].memory_displacement == 0x30
            and _is_test_reg_self(insns[15], Code.TEST_RM64_R64, Register.R10)
            and _is_branch(insns[16], Code.JE_REL8_64, insns[19].ip)
            and insns[17].code == Code.INC_RM32 and insns[17].has_lock_prefix
            and insns[17].op0_kind == OpKind.MEMORY and insns[17].memory_base == Register.R10 and insns[17].memory_displacement == 8
            and _is_mem_r64(insns[18], Code.MOV_R64_RM64, 0, Register.R10, Register.RBX, 8)
            and _is_mem_r64(insns[19], Code.MOV_R64_RM64, 0, Register.RAX, Register.RBX, 0)
            and _is_mem_r64(insns[20], Code.LEA_R64_M, 0, Register.RDX, Register.RSP, 0x30)):
        return None
    if (_is_mem_r64(insns[21], Code.MOV_R64_RM64, 0, Register.R9, Register.RSI, 0)
            and _is_mem_r64(insns[22], Code.MOV_R64_RM64, 0, Register.R8, Register.R14, 0)):
        deref_a3 = True
    elif (_is_mem_r64(insns[21], Code.MOV_R64_RM64, 0, Register.R8, Register.R14, 0)
            and _is_mov_reg_reg(insns[22], Register.R9, Register.RSI)):
        deref_a3 = False
    else:
        return None
    if not (_is_stack_save(insns[23], Register.RAX, 0x30)
            and _is_stack_save(insns[24], Register.R10, 0x38)
            and _is_stack_save(insns[25], Register.RDI, 0x20)
            and _is_stack_load(insns[27], Register.RBX, 0x58)
            and _is_stack_load(insns[28], Register.RSI, 0x60)
            and _is_stack_load(insns[29], Register.RDI, 0x68)
            and _is_add_rsp(insns[30], 0x40)
            and _is_pop(insns[31], Register.R14)
            and insns[32].code == Code.RETNQ):
        return None
    return Member(
        rva=rva, size=size, family=ALLOC_SHARED_5,
        alloc_fn=alloc_fn, alloc_bytes=alloc_bytes,
        callee=ctor_fn, deref_a2=True, deref_a3=deref_a3,
    )


def _classify_alloc_tail_2(insns, rva, size):
    """51B custom-alloc + 2-arg tail-call construct (direct or deref a1) (14 instructions)."""
    from iced_x86 import Code, Register
    if size != 51 or len(insns) != 14:
        return None
    if not (_is_stack_save(insns[0], Register.RBX, 8)
            and _is_push(insns[1], Register.RDI)
            and _is_sub_rsp(insns[2], 0x20)
            and _is_mov_reg_reg(insns[3], Register.RDI, Register.RCX)
            and _is_mov_reg_reg(insns[4], Register.RBX, Register.RDX)):
        return None
    alloc_bytes = _is_mov_r32_imm(insns[5], Register.ECX)
    alloc_fn = _is_call_rip_indirect(insns[6])
    ctor_fn = _jmp_target(insns[13])
    if alloc_bytes is None or alloc_bytes == 0 or alloc_fn is None or ctor_fn is None:
        return None
    if (_is_mem_r64(insns[7], Code.MOV_R64_RM64, 0, Register.RDX, Register.RDI, 0)
            and _is_mov_reg_reg(insns[8], Register.R8, Register.RBX)):
        deref_a1 = True
    elif (_is_mov_reg_reg(insns[7], Register.R8, Register.RBX)
            and _is_mov_reg_reg(insns[8], Register.RDX, Register.RDI)):
        deref_a1 = False
    else:
        return None
    if not (_is_mov_reg_reg(insns[9], Register.RCX, Register.RAX)
            and _is_stack_load(insns[10], Register.RBX, 0x30)
            and _is_add_rsp(insns[11], 0x20)
            and _is_pop(insns[12], Register.RDI)):
        return None
    return Member(
        rva=rva, size=size, family=ALLOC_TAIL_2,
        alloc_fn=alloc_fn, alloc_bytes=alloc_bytes,
        callee=ctor_fn, deref_a1=deref_a1,
    )


def _classify_alloc_deref_u32_tail_3(insns, rva, size):
    """66B custom-alloc + deref u32 a1 + 3-arg tail-call construct (18 instructions)."""
    from iced_x86 import Code, OpKind, Register
    if size != 66 or len(insns) != 18:
        return None
    if not (_is_stack_save(insns[0], Register.RBX, 8)
            and _is_stack_save(insns[1], Register.RSI, 0x10)
            and _is_push(insns[2], Register.RDI)
            and _is_sub_rsp(insns[3], 0x20)
            and _is_mov_reg_reg(insns[4], Register.RSI, Register.RCX)
            and _is_mov_reg_reg(insns[5], Register.RBX, Register.R8)):
        return None
    alloc_bytes = _is_mov_r32_imm(insns[6], Register.ECX)
    alloc_fn = _is_call_rip_indirect(insns[8])
    ctor_fn = _jmp_target(insns[17])
    if alloc_bytes is None or alloc_bytes == 0 or alloc_fn is None or ctor_fn is None:
        return None
    if not (_is_mov_reg_reg(insns[7], Register.RDI, Register.RDX)
            and insns[9].code == Code.MOV_R32_RM32 and insns[9].op0_register == Register.EDX
            and insns[9].op1_kind == OpKind.MEMORY and insns[9].memory_base == Register.RSI
            and insns[9].memory_index == Register.NONE and insns[9].memory_displacement == 0
            and _is_mov_reg_reg(insns[10], Register.R9, Register.RBX)
            and _is_mov_reg_reg(insns[11], Register.RCX, Register.RAX)
            and _is_mov_reg_reg(insns[12], Register.R8, Register.RDI)
            and _is_stack_load(insns[13], Register.RBX, 0x30)
            and _is_stack_load(insns[14], Register.RSI, 0x38)
            and _is_add_rsp(insns[15], 0x20)
            and _is_pop(insns[16], Register.RDI)):
        return None
    return Member(
        rva=rva, size=size, family=ALLOC_DEREF_U32_TAIL_3,
        alloc_fn=alloc_fn, alloc_bytes=alloc_bytes,
        callee=ctor_fn,
    )


def _classify_alloc_deref2_5(insns, rva, size):
    """86B custom-alloc + 5-arg construct (direct a1, deref a2 & a3, direct a4) (23 instructions)."""
    from iced_x86 import Code, Register
    if size != 86 or len(insns) != 23:
        return None
    if not (_is_stack_save(insns[0], Register.RBX, 8)
            and _is_stack_save(insns[1], Register.RBP, 0x10)
            and _is_stack_save(insns[2], Register.RSI, 0x18)
            and _is_push(insns[3], Register.RDI)
            and _is_sub_rsp(insns[4], 0x30)
            and _is_mov_reg_reg(insns[5], Register.RBP, Register.RCX)
            and _is_mov_reg_reg(insns[6], Register.RBX, Register.R9)):
        return None
    alloc_bytes = _is_mov_r32_imm(insns[7], Register.ECX)
    alloc_fn = _is_call_rip_indirect(insns[10])
    ctor_fn = _call_target(insns[16])
    if alloc_bytes is None or alloc_bytes == 0 or alloc_fn is None or ctor_fn is None:
        return None
    if not (_is_mov_reg_reg(insns[8], Register.RDI, Register.R8)
            and _is_mov_reg_reg(insns[9], Register.RSI, Register.RDX)
            and _is_mem_r64(insns[11], Code.MOV_R64_RM64, 0, Register.R9, Register.RDI, 0)
            and _is_mov_reg_reg(insns[12], Register.RDX, Register.RBP)
            and _is_mem_r64(insns[13], Code.MOV_R64_RM64, 0, Register.R8, Register.RSI, 0)
            and _is_mov_reg_reg(insns[14], Register.RCX, Register.RAX)
            and _is_stack_save(insns[15], Register.RBX, 0x20)
            and _is_stack_load(insns[17], Register.RBX, 0x40)
            and _is_stack_load(insns[18], Register.RBP, 0x48)
            and _is_stack_load(insns[19], Register.RSI, 0x50)
            and _is_add_rsp(insns[20], 0x30)
            and _is_pop(insns[21], Register.RDI)
            and insns[22].code == Code.RETNQ):
        return None
    return Member(
        rva=rva, size=size, family=ALLOC_DEREF2_5,
        alloc_fn=alloc_fn, alloc_bytes=alloc_bytes,
        callee=ctor_fn,
    )


def _match_guarded_free_check(insns, idx, skip_ip):
    """Match: cmp qword ptr [alloc_fn],0; je skip_ip; mov rax,[free_fn]; test rax,rax; je skip_ip."""
    from iced_x86 import Code, OpKind, Register
    if idx + 5 > len(insns):
        return None
    i0 = insns[idx]
    if not (i0.code == Code.CMP_RM64_IMM8 and i0.op0_kind == OpKind.MEMORY
            and i0.memory_base == Register.RIP and i0.immediate8to64 == 0):
        return None
    alloc_fn = i0.ip_rel_memory_address
    if not _is_branch(insns[idx + 1], Code.JE_REL8_64, skip_ip):
        return None
    free_fn = _is_mov_r64_rip(insns[idx + 2], Register.RAX)
    if free_fn is None:
        return None
    if not (_is_test_reg_self(insns[idx + 3], Code.TEST_RM64_R64, Register.RAX)
            and _is_branch(insns[idx + 4], Code.JE_REL8_64, skip_ip)):
        return None
    return (alloc_fn, free_fn)


def _classify_buffer_clear_free(insns, rva, size):
    """60B guarded buffer free + zero 3 pointers (19 instructions)."""
    from iced_x86 import Code, OpKind, Register
    if size != 60 or len(insns) != 19:
        return None
    epi_ip = insns[16].ip
    zero_ip = insns[12].ip
    if not (_is_push(insns[0], Register.RBX)
            and _is_sub_rsp(insns[1], 0x20)
            and _is_mov_reg_reg(insns[2], Register.RBX, Register.RCX)
            and _is_mem_r64(insns[3], Code.MOV_R64_RM64, 0, Register.RCX, Register.RCX, 0)
            and _is_test_reg_self(insns[4], Code.TEST_RM64_R64, Register.RCX)
            and _is_branch(insns[5], Code.JE_REL8_64, epi_ip)):
        return None
    gf = _match_guarded_free_check(insns, 6, zero_ip)
    if gf is None:
        return None
    alloc_fn, free_fn = gf
    if not (insns[11].code == Code.CALL_RM64 and insns[11].op0_kind == OpKind.REGISTER and insns[11].op0_register == Register.RAX
            and _is_xor_r32_self(insns[12], Register.EAX)
            and _is_mem_r64(insns[13], Code.MOV_RM64_R64, 1, Register.RAX, Register.RBX, 0)
            and _is_mem_r64(insns[14], Code.MOV_RM64_R64, 1, Register.RAX, Register.RBX, 8)
            and _is_mem_r64(insns[15], Code.MOV_RM64_R64, 1, Register.RAX, Register.RBX, 0x10)
            and _is_add_rsp(insns[16], 0x20)
            and _is_pop(insns[17], Register.RBX)
            and insns[18].code == Code.RETNQ):
        return None
    return Member(
        rva=rva, size=size, family=BUFFER_CLEAR_FREE,
        alloc_fn=alloc_fn, free_fn=free_fn,
    )


def _classify_buffer_destroy_clear_free(insns, rva, size):
    """75B destroy range + guarded buffer free + zero 3 pointers (23 instructions)."""
    from iced_x86 import Code, OpKind, Register
    if size != 75 or len(insns) != 23:
        return None
    epi_ip = insns[20].ip
    zero_ip = insns[16].ip
    if not (_is_push(insns[0], Register.RBX)
            and _is_sub_rsp(insns[1], 0x20)
            and _is_mov_reg_reg(insns[2], Register.RBX, Register.RCX)
            and _is_mem_r64(insns[3], Code.MOV_R64_RM64, 0, Register.RCX, Register.RCX, 0)
            and _is_test_reg_self(insns[4], Code.TEST_RM64_R64, Register.RCX)
            and _is_branch(insns[5], Code.JE_REL8_64, epi_ip)
            and _is_mem_r64(insns[6], Code.MOV_R64_RM64, 0, Register.RDX, Register.RBX, 8)
            and _is_mov_reg_reg(insns[7], Register.R8, Register.RBX)):
        return None
    destroy_fn = _call_target(insns[8])
    if destroy_fn is None:
        return None
    gf = _match_guarded_free_check(insns, 9, zero_ip)
    if gf is None:
        return None
    alloc_fn, free_fn = gf
    if not (_is_mem_r64(insns[14], Code.MOV_R64_RM64, 0, Register.RCX, Register.RBX, 0)
            and insns[15].code == Code.CALL_RM64 and insns[15].op0_kind == OpKind.REGISTER and insns[15].op0_register == Register.RAX
            and _is_xor_r32_self(insns[16], Register.EAX)
            and _is_mem_r64(insns[17], Code.MOV_RM64_R64, 1, Register.RAX, Register.RBX, 0)
            and _is_mem_r64(insns[18], Code.MOV_RM64_R64, 1, Register.RAX, Register.RBX, 8)
            and _is_mem_r64(insns[19], Code.MOV_RM64_R64, 1, Register.RAX, Register.RBX, 0x10)
            and _is_add_rsp(insns[20], 0x20)
            and _is_pop(insns[21], Register.RBX)
            and insns[22].code == Code.RETNQ):
        return None
    return Member(
        rva=rva, size=size, family=BUFFER_DESTROY_CLEAR_FREE,
        alloc_fn=alloc_fn, free_fn=free_fn, callee=destroy_fn,
    )


def _classify_buffer_reset_free(insns, rva, size):
    """104B guarded buffer free + assign new (begin, begin+size*stride, begin+cap*stride) (29 instructions)."""
    from iced_x86 import Code, OpKind, Register
    if size != 104 or len(insns) != 29:
        return None
    assign_ip = insns[18].ip
    if not (_is_stack_save(insns[0], Register.RBX, 8)
            and _is_stack_save(insns[1], Register.RBP, 0x10)
            and _is_stack_save(insns[2], Register.RSI, 0x18)
            and _is_push(insns[3], Register.RDI)
            and _is_sub_rsp(insns[4], 0x20)
            and _is_mov_reg_reg(insns[5], Register.RBX, Register.RCX)
            and _is_mov_reg_reg(insns[6], Register.RSI, Register.R9)
            and _is_mem_r64(insns[7], Code.MOV_R64_RM64, 0, Register.RCX, Register.RCX, 0)
            and _is_mov_reg_reg(insns[8], Register.RBP, Register.R8)
            and _is_mov_reg_reg(insns[9], Register.RDI, Register.RDX)
            and _is_test_reg_self(insns[10], Code.TEST_RM64_R64, Register.RCX)
            and _is_branch(insns[11], Code.JE_REL8_64, assign_ip)):
        return None
    gf = _match_guarded_free_check(insns, 12, assign_ip)
    if gf is None:
        return None
    alloc_fn, free_fn = gf
    i18, i22 = insns[18], insns[22]
    if not (insns[17].code == Code.CALL_RM64 and insns[17].op0_kind == OpKind.REGISTER and insns[17].op0_register == Register.RAX
            and i18.code == Code.LEA_R64_M and i18.op0_register == Register.RAX
            and i18.memory_base == Register.RDI and i18.memory_index == Register.RBP and i18.memory_displacement == 0
            and i22.code == Code.LEA_R64_M and i22.op0_register == Register.RAX
            and i22.memory_base == Register.RDI and i22.memory_index == Register.RSI and i22.memory_displacement == 0
            and i18.memory_index_scale == i22.memory_index_scale
            and i18.memory_index_scale in (1, 2, 4, 8)
            and _is_mem_r64(insns[19], Code.MOV_RM64_R64, 1, Register.RDI, Register.RBX, 0)
            and _is_stack_load(insns[20], Register.RBP, 0x38)
            and _is_mem_r64(insns[21], Code.MOV_RM64_R64, 1, Register.RAX, Register.RBX, 8)
            and _is_stack_load(insns[23], Register.RSI, 0x40)
            and _is_mem_r64(insns[24], Code.MOV_RM64_R64, 1, Register.RAX, Register.RBX, 0x10)
            and _is_stack_load(insns[25], Register.RBX, 0x30)
            and _is_add_rsp(insns[26], 0x20)
            and _is_pop(insns[27], Register.RDI)
            and insns[28].code == Code.RETNQ):
        return None
    return Member(
        rva=rva, size=size, family=BUFFER_RESET_FREE,
        alloc_fn=alloc_fn, free_fn=free_fn, elem_stride=i18.memory_index_scale,
    )


_DISPATCH = {
    241: (_classify_sbo_payload_binder,),
    190: (_classify_sbo_empty_binder,),
    53: (_classify_sbo_payload_cloner,),
    123: (_classify_guarded_log_emit,),
    173: (_classify_null_log_forward,),
    469: (_classify_json_field_extract,),
    257: (_classify_json_callback_extract,),
    110: (_classify_alloc_shared_4,),
    130: (_classify_alloc_shared_5,),
    51: (_classify_alloc_tail_2,),
    66: (_classify_alloc_deref_u32_tail_3,),
    86: (_classify_alloc_deref2_5,),
    60: (_classify_buffer_clear_free,),
    75: (_classify_buffer_destroy_clear_free,),
    104: (_classify_buffer_reset_free,),
}


def classify(code, rva):
    """Classify one non-EH candidate function into one of the 15 BNet runtime wrapper shapes."""
    from iced_x86 import Decoder
    size = len(code)
    handlers = _DISPATCH.get(size)
    if not handlers:
        return None
    insns = list(Decoder(64, code, ip=rva))
    if any(ins.is_invalid for ins in insns):
        return None
    for handler in handlers:
        member = handler(insns, rva, size)
        if member is not None:
            return member
    return None


def _has_exception_handler(image, pdata_entries, rva, size):
    """Reject any function whose .pdata unwind entries declare an exception or unwind handler."""
    end = rva + size
    for begin, p_end, uw_rva in pdata_entries:
        if begin < end and rva < p_end:
            try:
                _, hdr = image.read(uw_rva, 4, 'unwind header')
            except Reject:
                return True
            if (hdr[0] >> 3) & 3:
                return True
    return False


def _is_rdata(image, target):
    try:
        flags = image.section_for(target).flags
    except Reject:
        return False
    return not (flags & (SCN_MEM_EXECUTE | SCN_MEM_WRITE))


def _is_writable_data(image, target):
    try:
        flags = image.section_for(target).flags
    except Reject:
        return False
    return bool(flags & SCN_MEM_WRITE) and not (flags & SCN_MEM_EXECUTE)


def _valid_member(image, member, starts):
    def check_callee(target):
        return leaf_family_batch.indexed_callee(
            image, SimpleNamespace(rva=member.rva, size=member.size, callee=target), starts
        )

    def check_code_ptr(target):
        if member.rva <= target < member.rva + member.size:
            return False
        if target not in starts and (target & 0xF) != 0:
            return False
        try:
            return bool(image.section_for(target).flags & SCN_MEM_EXECUTE)
        except Reject:
            return False

    for fn in (member.callee, member.callee2, member.callee3,
               member.log_filter, member.log_console_en, member.log_verbosity,
               member.log_console, member.log_sink_en, member.log_sink):
        if fn is not None and not check_callee(fn):
            return False
    for cl in (member.sbo_cleanup, member.heap_cleanup):
        if cl is not None and not check_code_ptr(cl):
            return False
    for ro in (member.base_vtable, member.derived_vtable, member.log_fmt, member.log_arg,
               member.fmt_missing, member.fmt_type, member.fmt_type_log,
               member.fmt_other, member.fmt_err_log):
        if ro is not None and not _is_rdata(image, ro):
            return False
    for rw in (member.alloc_fn, member.free_fn, member.log_channel):
        if rw is not None and not _is_writable_data(image, rw):
            return False
    return True


def _shard(members):
    out = []
    by_family = {}
    for m in members:
        by_family.setdefault(m.family, []).append(m)
    for base in BASE_FAMILIES:
        rows = by_family.get(base, [])
        if len(rows) <= SHARD_SIZE:
            out.extend(rows)
        else:
            for idx in range(0, len(rows), SHARD_SIZE):
                shard_name = f'{base}{idx // SHARD_SIZE + 1}'
                out.extend(replace(m, family=shard_name) for m in rows[idx:idx + SHARD_SIZE])
    return sorted(out, key=lambda m: m.rva)


def discover(image, extents, inventory):
    """Discover all validated non-EH members of the 15 BNet runtime wrapper families."""
    pdata_entries = list(image.runtime_functions())
    starts = set(extents) | {begin for begin, _, _ in pdata_entries}
    # Index .pdata by begin RVA for O(log N) overlap check
    pdata_sorted = sorted(pdata_entries)
    import bisect
    pdata_begins = [b for b, _, _ in pdata_sorted]

    def has_eh(rva, size):
        end = rva + size
        hi = bisect.bisect_left(pdata_begins, end)
        lo = max(0, bisect.bisect_left(pdata_begins, rva) - 8)
        return _has_exception_handler(image, pdata_sorted[lo:hi], rva, size)

    accepted = []
    for member in leaf_family_batch.candidates(image, extents, inventory, CANDIDATE_SIZES, classify):
        if not has_eh(member.rva, member.size) and _valid_member(image, member, starts):
            accepted.append(member)
    return _shard(accepted)


def _mangled_symbol(family, member):
    bf = member.base_family
    n = member.name
    if bf == SBO_PAYLOAD_BINDER:
        return f'?{n}@{family}@@SA?AU{family}SboBinding@@PEAPEAXPEAU{family}PayloadBox@@_K@Z'
    if bf == SBO_EMPTY_BINDER:
        return f'?{n}@{family}@@SA?AU{family}SboBinding@@PEAXPEAU{family}EmptyBox@@_K@Z'
    if bf == SBO_PAYLOAD_CLONER:
        return f'?{n}@{family}@@SAPEAU{family}PayloadBox@@PEAPEAX@Z'
    if bf == GUARDED_LOG_EMIT:
        return f'?{n}@{family}@@SAXPEAX_NPEBD@Z'
    if bf == NULL_LOG_FORWARD:
        return f'?{n}@{family}@@SAXPEAXPEBD@Z'
    if bf == JSON_FIELD_EXTRACT:
        return f'?{n}@{family}@@SA_NPEAU{family}JsonReader@@PEBDPEAX_N3@Z'
    if bf == JSON_CALLBACK_EXTRACT:
        return (f'?{n}@{family}@@SA_NPEAU{family}JsonReader@@PEBD'
                f'P6AXPEAX2PEAU{family}JsonErrorState@@2@Z2@Z')
    if bf == ALLOC_SHARED_4:
        a2 = 'PEAPEAX' if member.deref_a2 else 'PEAX'
        a3 = 'PEAX' if member.deref_a2 else '1'
        return f'?{n}@{family}@@SAPEAXPEAU{family}SharedRef@@{a2}{a3}@Z'
    if bf == ALLOC_SHARED_5:
        if member.deref_a3:
            return f'?{n}@{family}@@SAPEAXPEAU{family}SharedRef@@PEAPEAX1PEAX@Z'
        return f'?{n}@{family}@@SAPEAXPEAU{family}SharedRef@@PEAPEAXPEAX2@Z'
    if bf == ALLOC_TAIL_2:
        if member.deref_a1:
            return f'?{n}@{family}@@SAPEAXPEAPEAXPEAX@Z'
        return f'?{n}@{family}@@SAPEAXPEAX0@Z'
    if bf == ALLOC_DEREF_U32_TAIL_3:
        return f'?{n}@{family}@@SAPEAXPEAIPEAX1@Z'
    if bf == ALLOC_DEREF2_5:
        return f'?{n}@{family}@@SAPEAXPEAXPEAPEAX10@Z'
    if bf in (BUFFER_CLEAR_FREE, BUFFER_DESTROY_CLEAR_FREE):
        return f'?{n}@{family}@@SAXPEAU{family}Vec3Ptr@@@Z'
    if bf == BUFFER_RESET_FREE:
        elem_code = {1: 'E', 2: 'G', 4: 'I', 8: '_K'}[member.elem_stride]
        return f'?{n}@{family}@@SAXPEAU{family}VecBuf{member.elem_stride}@@PEA{elem_code}_K2@Z'
    raise ValueError(f'unexpected family: {family}')


def _log_block(indent, ch, lvl, f_filter, f_cen, f_verb, f_con, f_sen, f_sink, fmt_expr, arg_exprs):
    args = ', '.join(arg_exprs)
    sp = ' ' * indent
    return [
        f'{sp}if (!LogFilter_{f_filter:x}(gLogChannel_{ch:x}, {lvl}))',
        f'{sp}{{',
        f'{sp}    if (LogConsoleEn_{f_cen:x}() && LogVerbosity_{f_verb:x}(gLogChannel_{ch:x}) <= {lvl})',
        f'{sp}    {{',
        f'{sp}        {f_con}({lvl}, {fmt_expr}, {args});',
        f'{sp}    }}',
        f'{sp}    if (LogSinkEn_{f_sen:x}({lvl}, gLogChannel_{ch:x}))',
        f'{sp}    {{',
        f'{sp}        {f_sink}({lvl}, {fmt_expr}, {args});',
        f'{sp}    }}',
        f'{sp}}}',
    ]


def emit(root, family, members, build_id, boundary_label):
    """Emit one self-contained C++ unit (.h, .cpp, task.json, symbols.json) for a family shard."""
    bf = members[0].base_family
    preamble = [
        '// Source exposure: mixed (pinned image and community function boundaries).',
        f'// Boundary source: {boundary_label}',
        '// Code origin: generated structural reconstruction of BNet runtime SBO, allocator, buffer, and guarded-log wrappers.',
        f'// {family}, helper structs, methods and referenced symbols: GUESS(low) STANDIN; original ownership and layouts unknown.',
        '// Adapted bodies: none',
    ]
    header = list(preamble) + ['#pragma once']
    if bf in (ALLOC_SHARED_4, ALLOC_SHARED_5):
        header.append('#include <intrin.h>')
    header.append('')

    # Emit unit-local typed structs (all prefixed with {family} to satisfy code-rules R1)
    if bf in (SBO_PAYLOAD_BINDER, SBO_EMPTY_BINDER):
        header += [
            f'struct {family}SboBinding',
            '{',
            '    void* instance;',
            '    void* cleanup;',
            '};',
            '',
        ]
    if bf in (SBO_PAYLOAD_BINDER, SBO_PAYLOAD_CLONER):
        header += [
            f'struct {family}PayloadBox',
            '{',
            '    void* table;',
            '    void* payload;',
            '};',
            '',
        ]
    if bf == SBO_EMPTY_BINDER:
        header += [
            f'struct {family}EmptyBox',
            '{',
            '    void* table;',
            '};',
            '',
        ]
    if bf in (JSON_FIELD_EXTRACT, JSON_CALLBACK_EXTRACT):
        header += [
            f'struct {family}JsonErrorState',
            '{',
            '    unsigned long long code;',
            '};',
            '',
        ]
    if bf == JSON_FIELD_EXTRACT:
        header += [
            f'struct {family}JsonErrorBuf',
            '{',
            '    unsigned char unk00[0x10];',
            '    char text[1];',
            '};',
            '',
            f'struct {family}JsonReader',
            '{',
            '    const char* name;',
            '    void* doc;',
            f'    {family}JsonErrorState* errState;',
            f'    {family}JsonErrorBuf* errBuf;',
            '};',
            '',
            f'struct {family}JsonNode',
            '{',
            '    unsigned long long low;',
            '    unsigned long long high;',
            '};',
            '',
        ]
    if bf == JSON_CALLBACK_EXTRACT:
        header += [
            f'struct {family}JsonReader',
            '{',
            '    const char* name;',
            '    void* doc;',
            f'    {family}JsonErrorState* errState;',
            '    void* errBuf;',
            '};',
            '',
            f'using {family}JsonCallback = void (*)(void* node, void* ctx, {family}JsonErrorState* errState, void* errBuf);',
            '',
        ]
    if bf in (ALLOC_SHARED_4, ALLOC_SHARED_5):
        header += [
            f'struct {family}RefControl',
            '{',
            '    unsigned char unk00[0x08];',
            '    volatile long uses; // volatile: atomic refcount increment target',
            '',
            '    void AddRef()',
            '    {',
            '        // Justification: atomic refcount increment matches MSVC shared_ptr control block',
            '        _InterlockedIncrement(&uses);',
            '    }',
            '};',
            '',
            f'struct {family}SharedBase',
            '{',
            '    void* ptr = nullptr;',
            f'    {family}RefControl* ctrl = nullptr;',
            '',
            f'    {family}SharedBase() = default;',
            '',
            f'    void CopyFrom(const {family}SharedBase& other)',
            '    {',
            '        if (other.ctrl != nullptr)',
            '        {',
            '            other.ctrl->AddRef();',
            '        }',
            '        ptr = other.ptr;',
            '        ctrl = other.ctrl;',
            '    }',
            '};',
            '',
            f'struct {family}SharedRef : {family}SharedBase',
            '{',
            f'    {family}SharedRef() = default;',
            f'    {family}SharedRef(const {family}SharedRef& other)',
            '    {',
            '        CopyFrom(other);',
            '    }',
            f'    ~{family}SharedRef();',
            '};',
            '',
        ]
    if bf in (BUFFER_CLEAR_FREE, BUFFER_DESTROY_CLEAR_FREE):
        header += [
            f'struct {family}Vec3Ptr',
            '{',
            '    void* begin;',
            '    void* end;',
            '    void* cap;',
            '};',
            '',
        ]
    if bf == BUFFER_RESET_FREE:
        elem_types = {1: 'unsigned char', 2: 'unsigned short', 4: 'unsigned int', 8: 'unsigned long long'}
        for stride in sorted({m.elem_stride for m in members}):
            et = elem_types[stride]
            header += [
                f'struct {family}VecBuf{stride}',
                '{',
                f'    {et}* begin;',
                f'    {et}* end;',
                f'    {et}* cap;',
                '};',
                '',
            ]

    # Collect unique external declarations and symbols.json entries
    symbols = {}
    externs = []

    def add_extern(name, decl, rva):
        if name not in symbols:
            symbols[name] = dict(rva=rva, size=None)
            externs.append(decl)

    for m in members:
        if m.alloc_fn is not None:
            add_extern(f'gAllocFn_{m.alloc_fn:x}',
                       f'extern "C" void* (*gAllocFn_{m.alloc_fn:x})(unsigned long long size);', m.alloc_fn)
        if m.free_fn is not None:
            add_extern(f'gFreeFn_{m.free_fn:x}',
                       f'extern "C" void (*gFreeFn_{m.free_fn:x})(void* ptr);', m.free_fn)
        if m.log_channel is not None:
            add_extern(f'gLogChannel_{m.log_channel:x}',
                       f'extern "C" unsigned int gLogChannel_{m.log_channel:x};', m.log_channel)
        if m.log_filter is not None:
            add_extern(f'LogFilter_{m.log_filter:x}',
                       f'extern "C" bool LogFilter_{m.log_filter:x}(unsigned int channel, int level);', m.log_filter)
        if m.log_console_en is not None:
            add_extern(f'LogConsoleEn_{m.log_console_en:x}',
                       f'extern "C" bool LogConsoleEn_{m.log_console_en:x}();', m.log_console_en)
        if m.log_verbosity is not None:
            add_extern(f'LogVerbosity_{m.log_verbosity:x}',
                       f'extern "C" int LogVerbosity_{m.log_verbosity:x}(unsigned int channel);', m.log_verbosity)
        if m.log_sink_en is not None:
            add_extern(f'LogSinkEn_{m.log_sink_en:x}',
                       f'extern "C" bool LogSinkEn_{m.log_sink_en:x}(int level, unsigned int channel);', m.log_sink_en)
        if m.log_console is not None:
            if bf in (SBO_PAYLOAD_BINDER, SBO_EMPTY_BINDER):
                add_extern(f'LogConsoleU64_{m.log_console:x}',
                           f'extern "C" void LogConsoleU64_{m.log_console:x}(int level, const char* fmt, unsigned long long arg);',
                           m.log_console)
            elif bf in (GUARDED_LOG_EMIT, NULL_LOG_FORWARD):
                add_extern(f'LogConsoleStr1_{m.log_console:x}',
                           f'extern "C" void LogConsoleStr1_{m.log_console:x}(int level, const char* fmt, const char* arg);',
                           m.log_console)
            elif bf == JSON_FIELD_EXTRACT:
                add_extern(f'LogConsoleStr1_{m.log_console:x}',
                           f'extern "C" void LogConsoleStr1_{m.log_console:x}(int level, const char* fmt, const char* arg);',
                           m.log_console)
                add_extern(f'LogConsoleStr2_{m.log_console:x}',
                           f'extern "C" void LogConsoleStr2_{m.log_console:x}(int level, const char* fmt, const char* a1, const char* a2);',
                           m.log_console)
            elif bf == JSON_CALLBACK_EXTRACT:
                add_extern(f'LogConsoleStr2_{m.log_console:x}',
                           f'extern "C" void LogConsoleStr2_{m.log_console:x}(int level, const char* fmt, const char* a1, const char* a2);',
                           m.log_console)
        if m.log_sink is not None:
            if bf in (SBO_PAYLOAD_BINDER, SBO_EMPTY_BINDER):
                add_extern(f'LogSinkU64_{m.log_sink:x}',
                           f'extern "C" void LogSinkU64_{m.log_sink:x}(int level, const char* fmt, unsigned long long arg);',
                           m.log_sink)
            elif bf in (GUARDED_LOG_EMIT, NULL_LOG_FORWARD):
                add_extern(f'LogSinkStr1_{m.log_sink:x}',
                           f'extern "C" void LogSinkStr1_{m.log_sink:x}(int level, const char* fmt, const char* arg);',
                           m.log_sink)
            elif bf == JSON_FIELD_EXTRACT:
                add_extern(f'LogSinkStr1_{m.log_sink:x}',
                           f'extern "C" void LogSinkStr1_{m.log_sink:x}(int level, const char* fmt, const char* arg);',
                           m.log_sink)
                add_extern(f'LogSinkStr2_{m.log_sink:x}',
                           f'extern "C" void LogSinkStr2_{m.log_sink:x}(int level, const char* fmt, const char* a1, const char* a2);',
                           m.log_sink)
            elif bf == JSON_CALLBACK_EXTRACT:
                add_extern(f'LogSinkStr2_{m.log_sink:x}',
                           f'extern "C" void LogSinkStr2_{m.log_sink:x}(int level, const char* fmt, const char* a1, const char* a2);',
                           m.log_sink)
        for vt in (m.base_vtable, m.derived_vtable):
            if vt is not None:
                add_extern(f'Table_{vt:x}', f'extern "C" unsigned char Table_{vt:x}[];', vt)
        for cl in (m.sbo_cleanup, m.heap_cleanup):
            if cl is not None:
                add_extern(f'Cleanup_{cl:x}', f'extern "C" unsigned char Cleanup_{cl:x}[];', cl)
        for s in (m.log_fmt, m.log_arg, m.fmt_missing, m.fmt_type, m.fmt_type_log, m.fmt_other, m.fmt_err_log):
            if s is not None:
                add_extern(f'Text_{s:x}', f'extern "C" const char Text_{s:x}[];', s)
        if bf == NULL_LOG_FORWARD:
            add_extern(f'StrLen_{m.callee:x}', f'extern "C" unsigned long long StrLen_{m.callee:x}(const char* str);', m.callee)
            add_extern(f'Forward_{m.callee2:x}', f'extern "C" void Forward_{m.callee2:x}(void* ctx, const char* str, unsigned long long len);', m.callee2)
        elif bf == JSON_FIELD_EXTRACT:
            add_extern(f'JsonLookup_{m.callee:x}',
                       f'extern "C" int JsonLookup_{m.callee:x}(void* doc, const char* key, {family}JsonNode* out);', m.callee)
            add_extern(f'JsonDecode_{m.callee2:x}',
                       f'extern "C" void JsonDecode_{m.callee2:x}({family}JsonNode* node, void* out);', m.callee2)
            add_extern(f'JsonFormatErr_{m.callee3:x}',
                       f'extern "C" void JsonFormatErr_{m.callee3:x}({family}JsonErrorBuf* buf, const char* fmt, const char* name, const char* key);', m.callee3)
        elif bf == JSON_CALLBACK_EXTRACT:
            add_extern(f'JsonLookup_{m.callee:x}',
                       f'extern "C" int JsonLookup_{m.callee:x}(void* doc, const char* key, void** outNode);', m.callee)
            add_extern(f'JsonFormatErr_{m.callee3:x}',
                       f'extern "C" void JsonFormatErr_{m.callee3:x}(void* buf, const char* fmt, const char* name, const char* key);', m.callee3)
        elif bf == ALLOC_SHARED_4:
            add_extern(f'Construct_{m.callee:x}',
                       f'extern "C" void* Construct_{m.callee:x}(void* mem, {family}SharedRef sp, void* a2, void* a3);', m.callee)
        elif bf == ALLOC_SHARED_5:
            add_extern(f'Construct_{m.callee:x}',
                       f'extern "C" void* Construct_{m.callee:x}(void* mem, {family}SharedRef sp, void* a2, void* a3, void* a4);', m.callee)
        elif bf == ALLOC_TAIL_2:
            add_extern(f'Construct_{m.callee:x}',
                       f'extern "C" void* Construct_{m.callee:x}(void* mem, void* a1, void* a2);', m.callee)
        elif bf == ALLOC_DEREF_U32_TAIL_3:
            add_extern(f'Construct_{m.callee:x}',
                       f'extern "C" void* Construct_{m.callee:x}(void* mem, unsigned int a1, void* a2, void* a3);', m.callee)
        elif bf == ALLOC_DEREF2_5:
            add_extern(f'Construct_{m.callee:x}',
                       f'extern "C" void* Construct_{m.callee:x}(void* mem, void* a1, void* a2, void* a3, void* a4);', m.callee)
        elif bf == BUFFER_DESTROY_CLEAR_FREE:
            add_extern(f'DestroyRange_{m.callee:x}',
                       f'extern "C" void DestroyRange_{m.callee:x}(void* begin, void* end, {family}Vec3Ptr* self);', m.callee)

    header.extend(externs)
    header += ['', f'struct {family}', '{']

    source = list(preamble)
    functions = []
    bodies = []

    elem_types = {1: 'unsigned char', 2: 'unsigned short', 4: 'unsigned int', 8: 'unsigned long long'}
    for m in members:
        sym = _mangled_symbol(family, m)
        symbols[sym] = dict(rva=m.rva, size=m.size)
        functions.append(dict(rva=m.rva, size=m.size, symbol=sym))
        source.append(f'// 0x{m.rva:x} {family}::{m.name}')

        if bf == SBO_PAYLOAD_BINDER:
            sig = f'{family}SboBinding {m.name}(void** src, {family}PayloadBox* buf, unsigned long long cap)'
            header.append(f'    static {sig};')
            bodies += [
                '',
                f'{family}SboBinding {family}::{m.name}(void** src, {family}PayloadBox* buf, unsigned long long cap)',
                '{',
                f'    {family}SboBinding result;',
                f'    if (cap >= {m.alloc_bytes}ull)',
                '    {',
                f'        buf->table = Table_{m.base_vtable:x};',
                '        buf->payload = *src;',
                f'        buf->table = Table_{m.derived_vtable:x};',
                '        result.instance = buf;',
                f'        result.cleanup = Cleanup_{m.sbo_cleanup:x};',
                '        return result;',
                '    }',
            ]
            bodies += _log_block(
                4, m.log_channel, m.log_level, m.log_filter, m.log_console_en, m.log_verbosity,
                f'LogConsoleU64_{m.log_console:x}', m.log_sink_en, f'LogSinkU64_{m.log_sink:x}',
                f'Text_{m.log_fmt:x}', [f'{m.alloc_bytes}ull'],
            )
            bodies += [
                f'    {family}PayloadBox* heap = static_cast<{family}PayloadBox*>(gAllocFn_{m.alloc_fn:x}({m.alloc_bytes}ull));',
                '    result.instance = heap;',
                f'    heap->table = Table_{m.base_vtable:x};',
                '    heap->payload = *src;',
                f'    heap->table = Table_{m.derived_vtable:x};',
                f'    result.cleanup = Cleanup_{m.heap_cleanup:x};',
                '    return result;',
                '}',
            ]
        elif bf == SBO_EMPTY_BINDER:
            sig = f'{family}SboBinding {m.name}(void* unused, {family}EmptyBox* buf, unsigned long long cap)'
            header.append(f'    static {sig};')
            bodies += [
                '',
                f'{family}SboBinding {family}::{m.name}(void* unused, {family}EmptyBox* buf, unsigned long long cap)',
                '{',
                '    (void)unused;',
                f'    {family}SboBinding result;',
                f'    if (cap >= {m.alloc_bytes}ull)',
                '    {',
                f'        buf->table = Table_{m.derived_vtable:x};',
                '        result.instance = buf;',
                f'        result.cleanup = Cleanup_{m.sbo_cleanup:x};',
                '        return result;',
                '    }',
            ]
            bodies += _log_block(
                4, m.log_channel, m.log_level, m.log_filter, m.log_console_en, m.log_verbosity,
                f'LogConsoleU64_{m.log_console:x}', m.log_sink_en, f'LogSinkU64_{m.log_sink:x}',
                f'Text_{m.log_fmt:x}', [f'{m.alloc_bytes}ull'],
            )
            bodies += [
                f'    {family}EmptyBox* heap = static_cast<{family}EmptyBox*>(gAllocFn_{m.alloc_fn:x}({m.alloc_bytes}ull));',
                '    result.instance = heap;',
                f'    heap->table = Table_{m.derived_vtable:x};',
                f'    result.cleanup = Cleanup_{m.heap_cleanup:x};',
                '    return result;',
                '}',
            ]
        elif bf == SBO_PAYLOAD_CLONER:
            sig = f'{family}PayloadBox* {m.name}(void** src)'
            header.append(f'    static {sig};')
            bodies += [
                '',
                f'{family}PayloadBox* {family}::{m.name}(void** src)',
                '{',
                f'    {family}PayloadBox* heap = static_cast<{family}PayloadBox*>(gAllocFn_{m.alloc_fn:x}({m.alloc_bytes}ull));',
                f'    heap->table = Table_{m.base_vtable:x};',
                '    heap->payload = *src;',
                f'    heap->table = Table_{m.derived_vtable:x};',
                '    return heap;',
                '}',
            ]
        elif bf == GUARDED_LOG_EMIT:
            sig = f'void {m.name}(void* unused, bool ok, const char* msg)'
            header.append(f'    static {sig};')
            bodies += [
                '',
                f'void {family}::{m.name}(void* unused, bool ok, const char* msg)',
                '{',
                '    (void)unused;',
                '    if (!ok)',
                '    {',
            ]
            bodies += _log_block(
                8, m.log_channel, m.log_level, m.log_filter, m.log_console_en, m.log_verbosity,
                f'LogConsoleStr1_{m.log_console:x}', m.log_sink_en, f'LogSinkStr1_{m.log_sink:x}',
                f'Text_{m.log_fmt:x}', ['msg'],
            )
            bodies += ['    }', '}']
        elif bf == NULL_LOG_FORWARD:
            sig = f'void {m.name}(void* ctx, const char* str)'
            header.append(f'    static {sig};')
            bodies += [
                '',
                f'void {family}::{m.name}(void* ctx, const char* str)',
                '{',
                '    if (str == nullptr)',
                '    {',
            ]
            bodies += _log_block(
                8, m.log_channel, m.log_level, m.log_filter, m.log_console_en, m.log_verbosity,
                f'LogConsoleStr1_{m.log_console:x}', m.log_sink_en, f'LogSinkStr1_{m.log_sink:x}',
                f'Text_{m.log_fmt:x}', [f'Text_{m.log_arg:x}'],
            )
            bodies += [
                '        return;',
                '    }',
                f'    Forward_{m.callee2:x}(ctx, str, StrLen_{m.callee:x}(str));',
                '}',
            ]
        elif bf == JSON_FIELD_EXTRACT:
            sig = f'bool {m.name}({family}JsonReader* reader, const char* key, void* out, bool quietErr, bool quietMissing)'
            header.append(f'    static {sig};')
            bodies += [
                '',
                f'bool {family}::{m.name}({family}JsonReader* reader, const char* key, void* out, bool quietErr, bool quietMissing)',
                '{',
                f'    {family}JsonNode node = {{ 0, 0 }};',
                f'    int rc = JsonLookup_{m.callee:x}(reader->doc, key, &node);',
                '    if (rc == 0)',
                '    {',
                f'        JsonDecode_{m.callee2:x}(&node, out);',
                '        return true;',
                '    }',
                f'    if (rc == {m.missing_rc})',
                '    {',
                '        if (!quietMissing)',
                '        {',
                f'            JsonFormatErr_{m.callee3:x}(reader->errBuf, Text_{m.fmt_missing:x}, reader->name, key);',
                f'            reader->errState->code = 0x{m.err_code_other:x}ull;',
            ]
            bodies += _log_block(
                12, m.log_channel, m.log_level, m.log_filter, m.log_console_en, m.log_verbosity,
                f'LogConsoleStr1_{m.log_console:x}', m.log_sink_en, f'LogSinkStr1_{m.log_sink:x}',
                f'Text_{m.fmt_err_log:x}', ['reader->errBuf->text'],
            )
            bodies += [
                '        }',
                '        return false;',
                '    }',
                '    if (!quietErr)',
                '    {',
                f'        if (rc == {m.type_rc})',
                '        {',
                f'            JsonFormatErr_{m.callee3:x}(reader->errBuf, Text_{m.fmt_type:x}, reader->name, key);',
                f'            reader->errState->code = 0x{m.err_code_type:x}ull;',
            ]
            bodies += _log_block(
                12, m.log_channel, m.log_level, m.log_filter, m.log_console_en, m.log_verbosity,
                f'LogConsoleStr2_{m.log_console:x}', m.log_sink_en, f'LogSinkStr2_{m.log_sink:x}',
                f'Text_{m.fmt_type_log:x}', ['reader->name', 'key'],
            )
            bodies += [
                '        }',
                '        else',
                '        {',
                f'            JsonFormatErr_{m.callee3:x}(reader->errBuf, Text_{m.fmt_other:x}, reader->name, key);',
                f'            reader->errState->code = 0x{m.err_code_other:x}ull;',
            ]
            bodies += _log_block(
                12, m.log_channel, m.log_level, m.log_filter, m.log_console_en, m.log_verbosity,
                f'LogConsoleStr1_{m.log_console:x}', m.log_sink_en, f'LogSinkStr1_{m.log_sink:x}',
                f'Text_{m.fmt_err_log:x}', ['reader->errBuf->text'],
            )
            bodies += [
                '        }',
                '    }',
                '    return false;',
                '}',
            ]
        elif bf == JSON_CALLBACK_EXTRACT:
            sig = f'bool {m.name}({family}JsonReader* reader, const char* key, {family}JsonCallback cb, void* ctx)'
            header.append(f'    static {sig};')
            bodies += [
                '',
                f'bool {family}::{m.name}({family}JsonReader* reader, const char* key, {family}JsonCallback cb, void* ctx)',
                '{',
                '    void* node;',
                f'    if (JsonLookup_{m.callee:x}(reader->doc, key, &node) == 0)',
                '    {',
                '        cb(node, ctx, reader->errState, reader->errBuf);',
                '        return reader->errState->code == 0;',
                '    }',
                f'    JsonFormatErr_{m.callee3:x}(reader->errBuf, Text_{m.fmt_type:x}, reader->name, key);',
                f'    reader->errState->code = 0x{m.err_code_type:x}ull;',
            ]
            bodies += _log_block(
                4, m.log_channel, m.log_level, m.log_filter, m.log_console_en, m.log_verbosity,
                f'LogConsoleStr2_{m.log_console:x}', m.log_sink_en, f'LogSinkStr2_{m.log_sink:x}',
                f'Text_{m.fmt_type_log:x}', ['reader->name', 'key'],
            )
            bodies += ['    return false;', '}']
        elif bf == ALLOC_SHARED_4:
            a2_ty = 'void**' if m.deref_a2 else 'void*'
            a2_expr = '*a2' if m.deref_a2 else 'a2'
            sig = f'void* {m.name}({family}SharedRef* sp, {a2_ty} a2, void* a3)'
            header.append(f'    static {sig};')
            bodies += [
                '',
                f'void* {family}::{m.name}({family}SharedRef* sp, {a2_ty} a2, void* a3)',
                '{',
                f'    void* mem = gAllocFn_{m.alloc_fn:x}(0x{m.alloc_bytes:x}ull);',
                f'    return Construct_{m.callee:x}(mem, *sp, {a2_expr}, a3);',
                '}',
            ]
        elif bf == ALLOC_SHARED_5:
            a3_ty = 'void**' if m.deref_a3 else 'void*'
            a3_expr = '*a3' if m.deref_a3 else 'a3'
            sig = f'void* {m.name}({family}SharedRef* sp, void** a2, {a3_ty} a3, void* a4)'
            header.append(f'    static {sig};')
            bodies += [
                '',
                f'void* {family}::{m.name}({family}SharedRef* sp, void** a2, {a3_ty} a3, void* a4)',
                '{',
                f'    void* mem = gAllocFn_{m.alloc_fn:x}(0x{m.alloc_bytes:x}ull);',
                f'    return Construct_{m.callee:x}(mem, *sp, *a2, {a3_expr}, a4);',
                '}',
            ]
        elif bf == ALLOC_TAIL_2:
            a1_ty = 'void**' if m.deref_a1 else 'void*'
            a1_expr = '*a1' if m.deref_a1 else 'a1'
            sig = f'void* {m.name}({a1_ty} a1, void* a2)'
            header.append(f'    static {sig};')
            bodies += [
                '',
                f'void* {family}::{m.name}({a1_ty} a1, void* a2)',
                '{',
                f'    void* mem = gAllocFn_{m.alloc_fn:x}(0x{m.alloc_bytes:x}ull);',
                f'    return Construct_{m.callee:x}(mem, {a1_expr}, a2);',
                '}',
            ]
        elif bf == ALLOC_DEREF_U32_TAIL_3:
            sig = f'void* {m.name}(unsigned int* a1, void* a2, void* a3)'
            header.append(f'    static {sig};')
            bodies += [
                '',
                f'void* {family}::{m.name}(unsigned int* a1, void* a2, void* a3)',
                '{',
                f'    void* mem = gAllocFn_{m.alloc_fn:x}(0x{m.alloc_bytes:x}ull);',
                f'    return Construct_{m.callee:x}(mem, *a1, a2, a3);',
                '}',
            ]
        elif bf == ALLOC_DEREF2_5:
            sig = f'void* {m.name}(void* a1, void** a2, void** a3, void* a4)'
            header.append(f'    static {sig};')
            bodies += [
                '',
                f'void* {family}::{m.name}(void* a1, void** a2, void** a3, void* a4)',
                '{',
                f'    void* mem = gAllocFn_{m.alloc_fn:x}(0x{m.alloc_bytes:x}ull);',
                f'    return Construct_{m.callee:x}(mem, a1, *a2, *a3, a4);',
                '}',
            ]
        elif bf == BUFFER_CLEAR_FREE:
            sig = f'void {m.name}({family}Vec3Ptr* vec)'
            header.append(f'    static {sig};')
            bodies += [
                '',
                f'void {family}::{m.name}({family}Vec3Ptr* vec)',
                '{',
                '    void* ptr = vec->begin;',
                '    if (ptr != nullptr)',
                '    {',
                f'        if (gAllocFn_{m.alloc_fn:x} != nullptr && gFreeFn_{m.free_fn:x} != nullptr)',
                '        {',
                f'            gFreeFn_{m.free_fn:x}(ptr);',
                '        }',
                '        vec->begin = nullptr;',
                '        vec->end = nullptr;',
                '        vec->cap = nullptr;',
                '    }',
                '}',
            ]
        elif bf == BUFFER_DESTROY_CLEAR_FREE:
            sig = f'void {m.name}({family}Vec3Ptr* vec)'
            header.append(f'    static {sig};')
            bodies += [
                '',
                f'void {family}::{m.name}({family}Vec3Ptr* vec)',
                '{',
                '    void* ptr = vec->begin;',
                '    if (ptr != nullptr)',
                '    {',
                f'        DestroyRange_{m.callee:x}(ptr, vec->end, vec);',
                f'        if (gAllocFn_{m.alloc_fn:x} != nullptr && gFreeFn_{m.free_fn:x} != nullptr)',
                '        {',
                f'            gFreeFn_{m.free_fn:x}(vec->begin);',
                '        }',
                '        vec->begin = nullptr;',
                '        vec->end = nullptr;',
                '        vec->cap = nullptr;',
                '    }',
                '}',
            ]
        elif bf == BUFFER_RESET_FREE:
            et = elem_types[m.elem_stride]
            sig = f'void {m.name}({family}VecBuf{m.elem_stride}* vec, {et}* newBuf, unsigned long long newSize, unsigned long long newCap)'
            header.append(f'    static {sig};')
            bodies += [
                '',
                f'void {family}::{m.name}({family}VecBuf{m.elem_stride}* vec, {et}* newBuf, unsigned long long newSize, unsigned long long newCap)',
                '{',
                f'    {et}* cur = vec->begin;',
                '    if (cur != nullptr)',
                '    {',
                f'        if (gAllocFn_{m.alloc_fn:x} != nullptr && gFreeFn_{m.free_fn:x} != nullptr)',
                '        {',
                f'            gFreeFn_{m.free_fn:x}(cur);',
                '        }',
                '    }',
                '    vec->begin = newBuf;',
                '    vec->end = newBuf + newSize;',
                '    vec->cap = newBuf + newCap;',
                '}',
            ]

    header.append('};')
    source.append(f'#include "{family}.h"')
    source.extend(bodies)
    return leaf_family_batch.write_unit(root, family, build_id, header, source, symbols, functions)


def main():
    return leaf_family_batch.run(__doc__, discover, emit)


if __name__ == '__main__':
    raise SystemExit(main())
