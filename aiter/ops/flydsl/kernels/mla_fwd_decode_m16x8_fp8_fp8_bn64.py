"""FlyDSL MLA decode kernel (nhead=128, fp8 Q, fp8 KV, bf16 output).

Transplanted from csrc/kernels/mla/hk/mi3xx_v32_fwd_decode_h128_fp8_fp8.cuh.
The gfx950 path from mi35x_v32_fwd_decode_m16x8_fp8_fp8.cuh is folded
into this module as an arch-dispatched branch in the same kernel.

Architecture: 8 warps / 512 threads, persistent-thread dispatch.
Default path: BLOCK_N=32, software V transpose through Vt LDS.
gfx950 path: BLOCK_N=64, V3 KV LDS layout, ds_read_b64_tr_b8 for V.

NOTE: Do NOT use ``from __future__ import annotations`` here -- it breaks
``fx.Constexpr`` detection in the FlyDSL AST rewriter.
"""
import threading

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, memref
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator

def _is_gfx950_arch(arch: str) -> bool:
    """Return True for the CDNA4 gfx950 kernel path."""
    return arch.lower().startswith('gfx950')
NUM_QO_HEADS: int = 128
NUM_KV_HEADS: int = 1
KV_LORA_RANK: int = 512
QK_NOPE_HEAD_DIM: int = KV_LORA_RANK
QK_ROPE_HEAD_DIM: int = 64
QK_HEAD_DIM: int = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM: int = KV_LORA_RANK
PAGE_SIZE: int = 1
NUM_WARPS: int = 8
WARP_SIZE: int = 64
NUM_THREADS: int = NUM_WARPS * WARP_SIZE
BLOCK_M: int = 128
IS_GFX950: bool = _is_gfx950_arch(get_hip_arch())
BLOCK_N: int = 64 if IS_GFX950 else 32
BLOCK_K: int = 32
TILE_M: int = BLOCK_M // NUM_WARPS
OCCUPANCY: int = 1
SIZE_MLA_WORK_INFO_IN_DW: int = 8
LOG2E: float = 1.4426950408889634
KV_NUM_COLS: int = 64
KV_NUM_BLOCKS: int = QK_HEAD_DIM // KV_NUM_COLS
KV_ROWS_PER_SUB: int = BLOCK_N // NUM_WARPS
KV_BYTES_PER_ROW: int = KV_NUM_COLS
KV_PAD_DW: int = 2
KV_SUB_BYTES: int = KV_ROWS_PER_SUB * KV_BYTES_PER_ROW + KV_PAD_DW * 4
KV_NUM_SUBS: int = BLOCK_N // KV_ROWS_PER_SUB
KV_BLOCK_BYTES: int = KV_SUB_BYTES * KV_NUM_SUBS
SZ_LDS_KV: int = KV_BLOCK_BYTES * KV_NUM_BLOCKS
VT_ROWS_PER_THR: int = 4
VT_COLS_PER_THR: int = 8
VT_ELEMS_PER_BLK: int = VT_ROWS_PER_THR * VT_COLS_PER_THR
VT_BLKS_PER_ROW: int = V_HEAD_DIM // VT_COLS_PER_THR
VT_BLKS_PER_ROW_PAD: int = VT_BLKS_PER_ROW + 2
VT_NUM_SUB_BLKS: int = 8
SZ_LDS_VT: int = VT_NUM_SUB_BLKS * (BLOCK_N // VT_NUM_SUB_BLKS * V_HEAD_DIM + 16 * 4)
Q_ELEM_PER_ROW: int = 64
Q_ELEM_PER_COL: int = 16
Q_PAD_BYTES_PER_2ROWS: int = 8
Q_BYTES_PER_2ROWS: int = Q_ELEM_PER_ROW * 2 + Q_PAD_BYTES_PER_2ROWS
SZ_LDS_Q_PER_WARP: int = Q_ELEM_PER_COL // 2 * Q_BYTES_PER_2ROWS
SZ_LDS_Q: int = NUM_WARPS * SZ_LDS_Q_PER_WARP
O16_NUM_ROWS: int = 16
O16_NUM_COLS: int = 32
O16_PAD_ELEM_PER_2ROWS: int = 4
O16_ELEM_PER_PAD_2ROWS: int = 2 * O16_NUM_COLS + O16_PAD_ELEM_PER_2ROWS
O16_LDS_PER_WARP: int = O16_NUM_ROWS // 2 * O16_ELEM_PER_PAD_2ROWS * 2
SZ_LDS_O16: int = NUM_WARPS * O16_LDS_PER_WARP
O32_NUM_ROWS: int = 16
O32_NUM_COLS: int = 32
O32_PAD_ELEM_PER_ROW: int = 4
O32_ELEM_PER_PAD_ROW: int = O32_NUM_COLS + O32_PAD_ELEM_PER_ROW
O32_LDS_PER_WARP: int = O32_NUM_ROWS * O32_ELEM_PER_PAD_ROW * 4
SZ_LDS_O32: int = NUM_WARPS * O32_LDS_PER_WARP
P_LDS_VT: int = 0
P_LDS_Q: int = SZ_LDS_VT
P_LDS_KV_0: int = P_LDS_Q + SZ_LDS_Q
P_LDS_KV_1: int = P_LDS_KV_0 + SZ_LDS_KV
V2_TOTAL_LDS_BYTES: int = P_LDS_KV_1 + SZ_LDS_KV
assert max(SZ_LDS_O16, SZ_LDS_O32) <= SZ_LDS_KV, 'Output LDS must fit in one KV buffer region'
MFMA_M: int = 16
MFMA_N: int = 16
MFMA_K: int = 32
MFMA_ELEM_PER_THR: int = MFMA_M * MFMA_K // WARP_SIZE
NUM_NOPE_ITERS: int = QK_NOPE_HEAD_DIM // (MFMA_K * 2)
NUM_ROPE_ITERS: int = QK_ROPE_HEAD_DIM // (MFMA_K * 2)
NUM_PV_ITERS: int = V_HEAD_DIM // (MFMA_N * 2)
V3_KV_SUB_BLOCK_ROWS: int = 4
V3_KV_SUB_BLOCK_COLS: int = 32
V3_KV_BYTES_PER_SUB_BLOCK: int = V3_KV_SUB_BLOCK_ROWS * V3_KV_SUB_BLOCK_COLS
V3_KV_BYTES_PER_2SUB_PADDED: int = V3_KV_BYTES_PER_SUB_BLOCK * 2 + KV_PAD_DW * 4
V3_KV_NUM_2SUBS: int = BLOCK_N // V3_KV_SUB_BLOCK_ROWS
V3_KV_BYTES_PER_BLOCK: int = V3_KV_BYTES_PER_2SUB_PADDED * V3_KV_NUM_2SUBS
V3_SZ_LDS_KV: int = V3_KV_BYTES_PER_BLOCK * KV_NUM_BLOCKS
V3_KV_NUM_ROWS_PER_WARP: int = V3_KV_SUB_BLOCK_ROWS * 2
V3_KV_NUM_COL_STRIPS: int = KV_NUM_COLS // V3_KV_SUB_BLOCK_COLS
V3_KV_NUM_WARPS_PER_COL: int = NUM_WARPS // V3_KV_NUM_COL_STRIPS
V3_KV_ROW_PASS_SLOT_STRIDE: int = V3_KV_NUM_WARPS_PER_COL * 2
V3_P_LDS_Q: int = 0
V3_P_LDS_KV_0: int = V3_P_LDS_Q + SZ_LDS_Q
V3_P_LDS_KV_1: int = V3_P_LDS_KV_0 + V3_SZ_LDS_KV
V3_TOTAL_LDS_BYTES: int = V3_P_LDS_KV_1 + V3_SZ_LDS_KV
assert max(SZ_LDS_O16, SZ_LDS_O32) <= V3_SZ_LDS_KV, 'Output LDS must fit in one gfx950 KV buffer region'
TOTAL_LDS_BYTES: int = V3_TOTAL_LDS_BYTES if IS_GFX950 else V2_TOTAL_LDS_BYTES

def _encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
    """Encode s_waitcnt bitfield for CDNA3 (gfx94x)."""
    vm_lo = vmcnt & 15
    vm_hi = vmcnt >> 4 & 3
    return vm_lo | expcnt << 4 | lgkmcnt << 8 | vm_hi << 14

def _barrier(vmcnt=63, lgkmcnt=63):
    """Emit s_waitcnt + s_barrier via inline asm."""
    parts = []
    needs_waitcnt = vmcnt < 63 or lgkmcnt < 63
    if needs_waitcnt:
        wc = []
        if vmcnt < 63:
            wc.append(f'vmcnt({vmcnt})')
        if lgkmcnt < 63:
            wc.append(f'lgkmcnt({lgkmcnt})')
        parts.append('s_waitcnt ' + ' '.join(wc))
    parts.append('s_barrier')
    _inline_asm_void([], '\n'.join(parts), '')

def _inline_asm_void(operands, asm_string, constraints):
    """Emit side-effecting void inline asm through the generated wrapper."""
    llvm.inline_asm(None, operands, asm_string, constraints, has_side_effects=True)
_LDS_PTR_TYPE = None

def _inttoptr_lds(byte_addr):
    """Convert an integer byte address to !llvm.ptr<3> (LDS pointer)."""
    global _LDS_PTR_TYPE
    if _LDS_PTR_TYPE is None:
        _LDS_PTR_TYPE = ir.Type.parse('!llvm.ptr<3>')
    return llvm.inttoptr(_LDS_PTR_TYPE, _raw(fx.Int64(byte_addr)))
_gep = buffer_ops.get_element_ptr

def _lds_load(byte_addr_index, vec_type, static_byte_offset=0):
    """LDS load via raw llvm.LoadOp on an LDS pointer (addr space 3)."""
    lds_ptr = _inttoptr_lds(byte_addr_index)
    if static_byte_offset != 0:
        lds_ptr = _gep(lds_ptr, static_byte_offset=static_byte_offset)
    return _ptr_load(vec_type, lds_ptr, alignment=16, nontemporal=True)

def _lds_load_volatile(base_i32, vec_type, byte_offset=0):
    """Volatile LDS load forcing ds_read_b64/b32 with immediate offset.

    Unlike _lds_load, uses volatile to prevent LLVM from merging adjacent
    loads into ds_read2 variants (which have limited 8-bit offsets).
    LLVM still tracks these as LDS loads for lgkmcnt.
    Input: base_i32 must be an i32 ir.Value (LDS byte address).
    """
    lds_ptr = _inttoptr_lds(ArithValue(base_i32).extui(T.i64))
    if byte_offset != 0:
        lds_ptr = _gep(lds_ptr, static_byte_offset=byte_offset)
    return _ptr_load(vec_type, lds_ptr, alignment=8, volatile_=True)

def _lds_ptr_from_i32(addr_i32, byte_offset=0):
    """Build an LDS pointer (ptr<3>) from an i32 byte address + optional static offset."""
    ptr = _inttoptr_lds(ArithValue(addr_i32).extui(T.i64))
    if byte_offset != 0:
        ptr = _gep(ptr, static_byte_offset=byte_offset)
    return ptr

def _ptr_load(result_type, ptr, *, alignment=None, volatile_=False, nontemporal=False):
    return llvm.LoadOp(result_type, ptr, alignment=alignment, volatile_=volatile_, nontemporal=nontemporal).result

def _ptr_store(value, ptr, *, alignment=None, volatile_=False):
    return llvm.StoreOp(_raw(value), ptr, alignment=alignment, volatile_=volatile_)

def _i32(value):
    """Cast index/ArithValue to i32.  No-op if already i32."""
    raw = _raw(value) if not isinstance(value, ir.Value) else value
    if raw.type == T.i32:
        return raw
    return _raw(fx.Int32(raw))

def _uniform_i32(value):
    """Cast to i32 and force a wave-uniform SGPR value for scalar inline asm operands."""
    return rocdl.readfirstlane(T.i32, _i32(value))

def _fast_exp2(val):
    """Bare v_exp_f32 via rocdl.exp2 -- no range reduction."""
    return rocdl.exp2(T.f32, _raw(val))

def _f32(val):
    """Convert Python/IR numeric values to a FlyDSL f32 wrapper."""
    if isinstance(val, fx.Float32):
        return val
    if isinstance(val, int):
        return fx.Float32(float(val))
    if isinstance(val, float):
        return fx.Float32(val)
    return fx.Float32(val)

def _idx(val):
    """Convert integer-like values to a FlyDSL index wrapper, preserving existing indexes."""
    if isinstance(val, fx.Index):
        return val
    return fx.Index(val)

def _pack_i32x2(lo, hi):
    """Pack two i32 values into a single i64: lo | (hi << 32)."""
    return _raw(ArithValue(lo).extui(T.i64) | ArithValue(hi).extui(T.i64) << 32)

@flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
def kn_mla_fwd_decode_m16x8_fp8_fp8(query: fx.Tensor, kv_buffer: fx.Tensor, kv_page_indices: fx.Tensor, work_indptr: fx.Tensor, work_info_set: fx.Tensor, final_output: fx.Tensor, split_output: fx.Tensor, split_lse: fx.Tensor, q_scale: fx.Tensor, kv_scale: fx.Tensor, softmax_scale: fx.Float32, num_qheads: fx.Constexpr[int]):
    """MLA decode forward kernel (nhead=128, fp8/fp8 -> bf16).

    Persistent-thread kernel: each workgroup picks up work items
    from ``work_indptr`` / ``work_info_set`` and processes them sequentially.
    """
    _STUB_EARLY_RETURN = False
    if const_expr(_STUB_EARLY_RETURN):
        return
    fm_fast = arith.FastMathFlags.fast
    fm_no_inf = arith.FastMathFlags.nnan | arith.FastMathFlags.nsz | arith.FastMathFlags.arcp | arith.FastMathFlags.contract | arith.FastMathFlags.afn | arith.FastMathFlags.reassoc

    def _mfma_fp8(result_type, operands, **kw):
        return rocdl.mfma_f32_16x16x32_fp8_fp8(result_type, operands, **kw)

    def _fadd(a, b, fastmath=fm_no_inf):
        return arith.addf(_raw(a), _raw(b), fastmath=fastmath)

    def _fsub(a, b, fastmath=fm_no_inf):
        return arith.subf(_raw(a), _raw(b), fastmath=fastmath)

    def _fmul(a, b, fastmath=fm_no_inf):
        return arith.mulf(_raw(a), _raw(b), fastmath=fastmath)

    def _fmax(a, b, fastmath=fm_no_inf):
        return arith.maximumf(_raw(a), _raw(b), fastmath=fastmath)
    arch = get_hip_arch()
    lds_allocator = SmemAllocator(None, arch=arch)
    lds_allocator.ptr = TOTAL_LDS_BYTES
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        lds_allocator.finalize()
    lds_buffer = lds_allocator.get_base()
    lds_base_idx = memref.extract_aligned_pointer_as_index(lds_buffer)
    c_perm0 = fx.Int32(83952640)
    c_perm1 = fx.Int32(117638658)
    c_perm2 = fx.Int32(84148480)
    c_perm3 = fx.Int32(117834498)

    def _vt_perm(src_hi, src_lo, sel):
        return rocdl.perm_b32(src_hi, src_lo, sel)
    c_neg_inf = fx.Float32(float('-inf'))
    c_zero_f32 = fx.Float32(0.0)
    c_one_f32 = fx.Float32(1.0)
    c_zero_i32 = fx.Int32(0)
    c_zero_v4f32 = Vec.filled(4, 0.0, fx.Float32)
    c_log2e = fx.Float32(LOG2E)
    c_inv_log2e = fx.Float32(1.0 / LOG2E)
    query_rsrc = buffer_ops.create_buffer_resource(query)
    kv_rsrc = buffer_ops.create_buffer_resource(kv_buffer)
    kv_page_indices_rsrc = buffer_ops.create_buffer_resource(kv_page_indices)
    work_indptr_rsrc = buffer_ops.create_buffer_resource(work_indptr)
    work_info_set_rsrc = buffer_ops.create_buffer_resource(work_info_set)
    final_output_rsrc = buffer_ops.create_buffer_resource(final_output)
    split_output_rsrc = buffer_ops.create_buffer_resource(split_output)
    split_lse_rsrc = buffer_ops.create_buffer_resource(split_lse)
    q_scale_rsrc = buffer_ops.create_buffer_resource(q_scale)
    kv_scale_rsrc = buffer_ops.create_buffer_resource(kv_scale)
    q_descale = buffer_ops.buffer_load(q_scale_rsrc, fx.Int32(0), vec_width=1, dtype=T.f32)
    kv_descale = buffer_ops.buffer_load(kv_scale_rsrc, fx.Int32(0), vec_width=1, dtype=T.f32)
    attention_scale = _fmul(_fmul(softmax_scale, q_descale, fm_fast), kv_descale, fm_fast)
    worker_idx = gpu.block_idx.x
    tid = gpu.thread_id('x')
    warp_idx = tid / WARP_SIZE
    lane_idx = tid % WARP_SIZE
    waves_per_query = num_qheads // TILE_M
    work_range = buffer_ops.buffer_load(work_indptr_rsrc, worker_idx, vec_width=2, dtype=T.i32)
    work_range_vec = Vec(work_range)
    work_start_i32 = rocdl.readfirstlane(T.i32, work_range_vec[0])
    work_end_i32 = rocdl.readfirstlane(T.i32, work_range_vec[1])
    work_start_idx = _idx(work_start_i32)
    work_end_idx = _idx(work_end_i32)
    if const_expr(IS_GFX950):
        kv_ld_row_base = warp_idx % V3_KV_NUM_WARPS_PER_COL * V3_KV_SUB_BLOCK_ROWS + lane_idx / 32 * 16 + lane_idx % 32 / 8
        kv_ld_col_base = _i32(warp_idx / V3_KV_NUM_WARPS_PER_COL * V3_KV_SUB_BLOCK_COLS + lane_idx % 8 * 4)
    else:
        kv_ld_row_base = lane_idx / 32 * 16 + lane_idx / 16 % 2 + warp_idx * 2
        kv_ld_col_base = _i32(lane_idx % 16 * 4)

    def _get_kv_ld_row(kv_tile_start_i32, kv_tile_end_i32, check_boundary, pass_idx=0):
        """Resolve physical KV row for this thread's assigned row.

        For OOB rows (row >= kv_end), returns -1 WITHOUT issuing a
        buffer_load -- avoids reading garbage from kv_page_indices.
        """
        row_idx = kv_ld_row_base + _idx(kv_tile_start_i32)
        if const_expr(IS_GFX950):
            row_idx = kv_ld_row_base + pass_idx * 32 + _idx(kv_tile_start_i32)
        if const_expr(check_boundary):
            phys_row = fx.Int32(-1)
            if row_idx < _idx(kv_tile_end_i32):
                phys_row = buffer_ops.buffer_load(kv_page_indices_rsrc, row_idx, vec_width=1, dtype=T.i32)
            return _raw(phys_row)
        else:
            return buffer_ops.buffer_load(kv_page_indices_rsrc, row_idx, vec_width=1, dtype=T.i32)

    def _async_load_k_tile(p_lds_kv_warp, row_i32, col_base_i32, block_idx_const, pass_idx=0, check_boundary=False):
        """Load one 32x64 block of KV data from VRAM to LDS.

        block_idx_const: Python int [0..8], which 64-col block.
        """
        if const_expr(IS_GFX950):
            lds_adjust = pass_idx * V3_KV_ROW_PASS_SLOT_STRIDE * V3_KV_BYTES_PER_2SUB_PADDED + block_idx_const * V3_KV_BYTES_PER_BLOCK - block_idx_const * KV_NUM_COLS
        else:
            lds_warp_offset = block_idx_const * KV_BLOCK_BYTES
            lds_adjust = lds_warp_offset - block_idx_const * KV_NUM_COLS
        lds_base_i32 = _i32(ArithValue(p_lds_kv_warp) + lds_adjust)

        def _emit_vram_to_lds():
            voff = _i32(ArithValue(row_i32) * QK_HEAD_DIM + col_base_i32)
            rocdl.buffer_load_to_lds(kv_rsrc, _lds_ptr_from_i32(lds_base_i32), voff, offset=block_idx_const * KV_NUM_COLS)
        if const_expr(check_boundary):
            is_oob = ArithValue(row_i32) == -1
            if is_oob:
                lds_addr = _i32(ArithValue(lds_base_i32) + block_idx_const * KV_NUM_COLS + _i32(lane_idx) * 4)
                lds_ptr = _lds_ptr_from_i32(lds_addr)
                _ptr_store(c_zero_i32, lds_ptr, alignment=4)
            else:
                _emit_vram_to_lds()
        else:
            _emit_vram_to_lds()

    def _async_load_kv_all(p_lds_kv_warp, row_p0_i32, col_base_i32, row_p1_i32=None, check_boundary=False):
        """Load all KV blocks of a tile.

        Pass-0 always runs. Pass-1 only runs on gfx950 (BLOCK_N=64), and
        `row_p1_i32` MUST be supplied in that case.
        """
        for blk in range_constexpr(KV_NUM_BLOCKS):
            _async_load_k_tile(p_lds_kv_warp, row_p0_i32, col_base_i32, blk, pass_idx=0, check_boundary=check_boundary)
        if const_expr(IS_GFX950):
            for blk in range_constexpr(KV_NUM_BLOCKS):
                _async_load_k_tile(p_lds_kv_warp, row_p1_i32, col_base_i32, blk, pass_idx=1, check_boundary=check_boundary)

    def _prefetch_k_tile_asm(p_lds_kv_warp, row_i32, col_base_i32, block_idx_const, pass_idx=0, check_boundary=True):
        """Prefetch one KV block via inline asm buffer_load_dword lds.

        Uses inline asm for BOTH the normal load AND the OOB zero-write
        so LLVM sees no LDS operations and won't insert spurious
        s_waitcnt vmcnt(0) before subsequent ds_read ops.

        check_boundary: controls OOB row==-1 check.
          - False (Python): skips check entirely -- caller guarantees valid row.
          - True (Python): always emits a branch on row==-1.
          - ir.Value (i1): emits a branch on check_boundary AND row==-1,
            allowing runtime bypass.
        """
        if const_expr(IS_GFX950):
            lds_adjust = pass_idx * V3_KV_ROW_PASS_SLOT_STRIDE * V3_KV_BYTES_PER_2SUB_PADDED + block_idx_const * V3_KV_BYTES_PER_BLOCK - block_idx_const * KV_NUM_COLS
        else:
            lds_adjust = block_idx_const * KV_BLOCK_BYTES - block_idx_const * KV_NUM_COLS
        lds_base_i32 = _i32(ArithValue(p_lds_kv_warp) + lds_adjust)

        def _emit_normal_load():
            voff = _i32(ArithValue(row_i32) * QK_HEAD_DIM + col_base_i32)
            col_off_imm = block_idx_const * KV_NUM_COLS
            lds_base_sgpr = _uniform_i32(lds_base_i32)
            asm_str = f's_mov_b32 m0, $0\ns_nop 0\nbuffer_load_dword $1, $2, 0 offen offset:{col_off_imm} lds'
            _inline_asm_void([lds_base_sgpr, voff, _raw(kv_rsrc)], asm_str, 's,v,s')
        if const_expr(check_boundary is False):
            _emit_normal_load()
        else:
            is_oob = ArithValue(row_i32) == -1
            if const_expr(check_boundary is not True):
                is_oob = _raw(ArithValue(check_boundary) & is_oob)
            if is_oob:
                lds_zero_addr = _i32(ArithValue(lds_base_i32) + block_idx_const * KV_NUM_COLS + _i32(lane_idx) * 4)
                _inline_asm_void([lds_zero_addr, _raw(c_zero_i32)], 'ds_write_b32 $0, $1', 'v,v')
            else:
                _emit_normal_load()
    if const_expr(IS_GFX950):
        k_row_mfma = lane_idx % MFMA_M
        k_col_mfma = lane_idx / MFMA_M * MFMA_ELEM_PER_THR
        k_lds_lane_offset = k_row_mfma / V3_KV_SUB_BLOCK_ROWS * V3_KV_BYTES_PER_2SUB_PADDED + k_row_mfma % V3_KV_SUB_BLOCK_ROWS * V3_KV_SUB_BLOCK_COLS + k_col_mfma
    else:
        k_row_in_mfma = lane_idx % MFMA_M
        k_row_phy = k_row_in_mfma / 2 * 4 + k_row_in_mfma % 2
        k_col_in_lane = lane_idx / MFMA_M * MFMA_ELEM_PER_THR
        k_lds_lane_offset = k_row_phy / 4 * KV_SUB_BYTES + k_row_phy % 4 * KV_BYTES_PER_ROW + k_col_in_lane % KV_NUM_COLS

    def _load_k_from_lds(k_base_i32, row_offset, col_offset):
        """Read 16x32 K sub-tile from LDS -> i64 for MFMA.

        row_offset: 0 or 16 (which half of BLOCK_N=32)
        col_offset: column offset in elements (multiple of 32)

        KvManagerV2 LDS address formula:
          row_phy = (row/2)*4 + (row%2)  where row = lane_idx % 16
          p = p_lds_kv + (row_phy/4)*KV_SUB_BYTES + (row_phy%4)*KV_BYTES_PER_ROW
              + (col%64)*sizeof(kv_t) + (col/64)*KV_BLOCK_BYTES
          fixed_offset = (row_offset/16)*2*KV_BYTES_PER_ROW
                       + (col_offset%64)*sizeof(kv_t)
                       + (col_offset/64)*KV_BLOCK_BYTES

        NOTE: The fixed_offset is passed via static_byte_offset so LLVM
        can potentially fold it into ds_read's immediate. Currently LLVM
        lowers this to ds_read2_b64 due to inttoptr; a proper fix needs
        FlyDSL infrastructure changes to emit ds_read_b64 with large offsets.
        """
        if const_expr(IS_GFX950):
            fixed_offset = row_offset // 32 * V3_KV_ROW_PASS_SLOT_STRIDE * V3_KV_BYTES_PER_2SUB_PADDED + row_offset % 32 // 16 * V3_KV_BYTES_PER_SUB_BLOCK + col_offset // KV_NUM_COLS * V3_KV_BYTES_PER_BLOCK + col_offset % KV_NUM_COLS // V3_KV_SUB_BLOCK_COLS * V3_KV_NUM_WARPS_PER_COL * V3_KV_BYTES_PER_2SUB_PADDED
        else:
            fixed_offset = row_offset // 16 * 2 * KV_BYTES_PER_ROW + col_offset % KV_NUM_COLS + col_offset // KV_NUM_COLS * KV_BLOCK_BYTES
        data = _lds_load_volatile(k_base_i32, T.i64, byte_offset=fixed_offset)
        return data

    def _load_v_from_lds(p_lds_kv_base_idx, warp_idx_val, lane_idx_val):
        """Load un-transposed V: each warp reads 16x128 region.

        KvManagerV2::load_v_to_gpr pattern:
          row = (warp%2)*16 + lane/16*4
          row_phy = ((row%16)/2)*4 + 2*(row/16) + (row%2)
          col = (lane%16)*8 + (warp/2)*128
        Returns 8 i32 values.
        """
        row = warp_idx_val % 2 * 16 + lane_idx_val / 16 * 4
        row_mod16 = row % 16
        row_phy = row_mod16 / 2 * 4 + 2 * (row / 16) + row % 2
        col = lane_idx_val % 16 * 8 + warp_idx_val / 2 * 128
        lds_v_offset = row_phy / 4 * KV_SUB_BYTES + row_phy % 4 * KV_BYTES_PER_ROW + col / KV_NUM_COLS * KV_BLOCK_BYTES + col % KV_NUM_COLS
        lds_addr = p_lds_kv_base_idx + lds_v_offset
        v_vals = []
        for pass_idx in range_constexpr(4):
            if const_expr(pass_idx == 0):
                off = 0
            elif const_expr(pass_idx == 1):
                off = KV_BYTES_PER_ROW
            elif const_expr(pass_idx == 2):
                off = KV_SUB_BYTES
            else:
                off = KV_SUB_BYTES + KV_BYTES_PER_ROW
            data = _lds_load(lds_addr, T.i32x2, static_byte_offset=off)
            data_vec = Vec(data)
            v_vals.append(data_vec[0])
            v_vals.append(data_vec[1])
        return v_vals

    def _transpose_v(v8):
        """12x v_perm_b32 to transpose 4x8 fp8 block.

        Ported from VtManagerV1::transpose_v.
        Input:  v8[0..7] in row-major 4x8 layout
        Output: v8[0..7] in transposed layout for Vt storage
        """
        t0_0 = _vt_perm(v8[2], v8[0], c_perm0)
        t2_0 = _vt_perm(v8[2], v8[0], c_perm1)
        t0_1 = _vt_perm(v8[3], v8[1], c_perm0)
        t2_1 = _vt_perm(v8[3], v8[1], c_perm1)
        t1_0 = _vt_perm(v8[6], v8[4], c_perm0)
        t3_0 = _vt_perm(v8[6], v8[4], c_perm1)
        t1_1 = _vt_perm(v8[7], v8[5], c_perm0)
        t3_1 = _vt_perm(v8[7], v8[5], c_perm1)
        r = [None] * 8
        r[0] = _vt_perm(t1_0, t0_0, c_perm2)
        r[1] = _vt_perm(t1_1, t0_1, c_perm2)
        r[2] = _vt_perm(t1_0, t0_0, c_perm3)
        r[3] = _vt_perm(t1_1, t0_1, c_perm3)
        r[4] = _vt_perm(t3_0, t2_0, c_perm2)
        r[5] = _vt_perm(t3_1, t2_1, c_perm2)
        r[6] = _vt_perm(t3_0, t2_0, c_perm3)
        r[7] = _vt_perm(t3_1, t2_1, c_perm3)
        return r

    def _store_vt_to_lds(vt_lds_base_idx, warp_idx_val, lane_idx_val, vt8):
        """VtManagerV1::store_transposed_v_to_lds.

        4x8 block-wise row-major layout, no padding between rows/cols.
        row_blk = (warp%2)*4 + lane/16
        col_blk = (lane%16) + (warp/2)*16
        block_offset = (row_blk * VT_BLKS_PER_ROW_PAD + col_blk) * VT_ELEMS_PER_BLK
        """
        row_blk = warp_idx_val % 2 * 4 + lane_idx_val / 16
        col_blk = lane_idx_val % 16 + warp_idx_val / 2 * 16
        block_offset = (row_blk * VT_BLKS_PER_ROW_PAD + col_blk) * VT_ELEMS_PER_BLK
        lds_vt_addr = vt_lds_base_idx + block_offset
        lo_packed = Vec.from_elements(vt8[0:4], fx.Int32)
        lo_i8 = Vec(lo_packed).bitcast(fx.Int8)
        lo_i8.store(lds_buffer, [lds_vt_addr])
        hi_packed = Vec.from_elements(vt8[4:8], fx.Int32)
        hi_i8 = Vec(hi_packed).bitcast(fx.Int8)
        hi_i8.store(lds_buffer, [lds_vt_addr + 16])

    def _load_vt_from_lds(vt_base_i32, col_offset):
        """VtManagerV1::load_transposed_v_to_gpr.

        Each warp reads 32x16 block from Vt LDS. Returns 2 i32 via ds_read_b32.
        vt_base_i32: i32 LDS byte address with lane offset pre-baked.
        col_offset: Python int, multiple of 16, in [0, 512).

        Lane offset pre-computed in vt_lds_lane_offset (top level).
        Only col_offset contributes a fixed immediate offset here.
        offset_tl_bl = 4 * VT_BLKS_PER_ROW_PAD * VT_ELEMS_PER_BLK = 8448
        """
        fixed_col_blk = col_offset // VT_COLS_PER_THR
        fixed_block_offset = fixed_col_blk * VT_ELEMS_PER_BLK
        offset_tl_bl = 4 * VT_BLKS_PER_ROW_PAD * VT_ELEMS_PER_BLK
        v0 = _lds_load_volatile(vt_base_i32, T.i32, byte_offset=fixed_block_offset)
        v1 = _lds_load_volatile(vt_base_i32, T.i32, byte_offset=fixed_block_offset + offset_tl_bl)
        return (v0, v1)

    def _shfl_xor_f32(val_f32, offset, width=WARP_SIZE):
        """XOR shuffle for f32 via bitcast to i32 and back."""
        val_i32 = _raw(ArithValue(val_f32).bitcast(T.i32))
        peer_i32 = ArithValue(val_i32).shuffle_xor(offset, width)
        return fx.Float32(ArithValue(peer_i32).bitcast(T.f32))

    def _warp_reduce_max_16(val):
        """Butterfly max reduce across MFMA column groups (strides 32, 16)."""
        w = _f32(val)
        for sh in [32, 16]:
            w = _fmax(w, _shfl_xor_f32(w, sh), fm_no_inf)
        return w

    def _warp_reduce_add_16(val):
        """Butterfly sum reduce across MFMA column groups (strides 32, 16)."""
        w = _f32(val)
        for sh in [32, 16]:
            w = w + _shfl_xor_f32(w, sh)
        return w
    p_lds_q_offset = V3_P_LDS_Q if IS_GFX950 else P_LDS_Q

    def _load_q_to_regs(qo_start_i32, query_wave_i32):
        """Load Q from VRAM to registers via LDS staging.

        QManagerV3: each warp loads 16x64 per pass, 9 passes total.
        VRAM -> LDS (ds_write_b128), then LDS -> register (ds_read_b64).
        Returns (q_nope_regs, q_rope_regs):
          q_nope_regs: list of 16 v2i64 (16 sub-tiles x 32 cols each)
          q_rope_regs: list of 2 v2i64 (2 sub-tiles x 32 cols each)
        """
        p_lds_q_warp = lds_base_idx + p_lds_q_offset + warp_idx * SZ_LDS_Q_PER_WARP
        s_offset = _idx(query_wave_i32) * (16 * QK_HEAD_DIM) + _idx(qo_start_i32) * (num_qheads * QK_HEAD_DIM)
        row = lane_idx / 4
        col = lane_idx % 4 * 16
        v_offset = row * QK_HEAD_DIM + col
        row_st = lane_idx / 4
        col_st = lane_idx % 4 * 16
        lds_st_offset = row_st / 2 * Q_BYTES_PER_2ROWS + row_st % 2 * Q_ELEM_PER_ROW + col_st
        row_ld = lane_idx % 16
        col_ld = lane_idx / 16 * 8
        lds_ld_offset = row_ld / 2 * Q_BYTES_PER_2ROWS + row_ld % 2 * Q_ELEM_PER_ROW + col_ld
        q_regs = []
        voff_dw = (v_offset + s_offset) // 4
        lds_st_addr = p_lds_q_warp + lds_st_offset
        lds_st_ptr = _inttoptr_lds(lds_st_addr)
        lds_rd_addr = p_lds_q_warp + lds_ld_offset

        def _q_buf_load(pass_idx):
            voff_pass = voff_dw + pass_idx * Q_ELEM_PER_ROW // 4
            return buffer_ops.buffer_load(query_rsrc, voff_pass, vec_width=4, dtype=T.i32)

        def _shuffle_q_through_lds(q_vram_data):
            """LDS write (ds_write_b128) + barrier + LDS read (2x ds_read_b64)."""
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            _ptr_store(q_vram_data, lds_st_ptr, alignment=16)
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            q0 = _lds_load(lds_rd_addr, T.i64, static_byte_offset=0)
            q1 = _lds_load(lds_rd_addr, T.i64, static_byte_offset=MFMA_K)
            return (q0, q1)
        loads = [None, None, None]
        loads[0] = _q_buf_load(0)
        loads[1] = _q_buf_load(1)
        for i in range_constexpr(9):
            slot = i % 3
            issue_pass = i + 2
            if const_expr(issue_pass < 9):
                rocdl.s_waitcnt(_encode_waitcnt(vmcnt=1))
                loads[issue_pass % 3] = _q_buf_load(issue_pass)
            else:
                rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
            q_regs.append(_shuffle_q_through_lds(loads[slot]))
        q_nope_packs = []
        for i in range_constexpr(8):
            q_nope_packs.append(q_regs[i][0])
            q_nope_packs.append(q_regs[i][1])
        q_rope_packs = [q_regs[8][0], q_regs[8][1]]
        return (q_nope_packs, q_rope_packs)
    P_VALS_PER_THR = BLOCK_N * MFMA_M // WARP_SIZE

    def _softmax_scale_p(p_vals, col_0_start, kv_end_i32, check_boundary):
        """Scale p_vals by softmax_scale, mask OOB to -inf.

        check_boundary: False (skip), True (always mask), or ir.Value i1
        (runtime: mask only when True at runtime).
        """
        result = [None] * P_VALS_PER_THR
        for i in range_constexpr(P_VALS_PER_THR):
            result[i] = _f32(p_vals[i]) * attention_scale
        if const_expr(check_boundary is not False):
            kv_end = _idx(kv_end_i32)
            for i in range_constexpr(P_VALS_PER_THR):
                sub_offset = i // 4 * 16 + i % 4
                pos = col_0_start + sub_offset
                is_oob = pos >= kv_end
                if const_expr(check_boundary is not True):
                    is_oob = _raw(ArithValue(check_boundary) & is_oob)
                result[i] = ArithValue(is_oob).select(_raw(c_neg_inf), result[i])
        return result

    def _softmax(p_vals, row_max_old, row_sum_e_old, is_first_iter, kv_tile_start_i32, kv_end_i32, check_boundary):
        """Online softmax: scale -> max -> exp2 -> sum -> rescale.

        p_vals: P_VALS_PER_THR f32 attention scores for this thread
        Returns: (p_exp_vals, row_max_new, row_sum_e_new, rescale)
        """
        col_0_start = lane_idx / 16 * 4 + _idx(kv_tile_start_i32)
        scaled = _softmax_scale_p(p_vals, col_0_start, kv_end_i32, check_boundary)
        local_max = scaled[0]
        for i in range_constexpr(1, P_VALS_PER_THR):
            local_max = _fmax(local_max, scaled[i], fm_no_inf)
        local_max = _warp_reduce_max_16(local_max)
        if const_expr(is_first_iter):
            new_row_max = local_max
            rescale = c_one_f32
        else:
            new_row_max = _fmax(local_max, row_max_old, fm_no_inf)
            diff = _fsub(row_max_old, new_row_max, fm_no_inf)
            rescale = _fast_exp2(_fmul(diff, c_log2e, fm_no_inf))
        p_exp_vals = [None] * P_VALS_PER_THR
        local_sum = c_zero_f32
        for i in range_constexpr(P_VALS_PER_THR):
            exp_arg = _fmul(_fsub(scaled[i], new_row_max, fm_no_inf), c_log2e, fm_no_inf)
            p_exp_vals[i] = _fast_exp2(exp_arg)
            local_sum = _fadd(local_sum, p_exp_vals[i], fm_no_inf)
        local_sum = _warp_reduce_add_16(local_sum)
        if const_expr(is_first_iter):
            row_sum_e_new = local_sum
        else:
            row_sum_e_new = _fadd(_f32(rescale) * row_sum_e_old, local_sum, fm_no_inf)
        return (p_exp_vals, new_row_max, row_sum_e_new, rescale)

    def _pack_p_to_fp8(p_exp_vals):
        """Pack softmax probabilities to fp8 for PV MFMA."""

        def _pack8(v):
            w0 = rocdl.cvt_pk_fp8_f32(T.i32, v[0], v[1], c_zero_i32, 0)
            w0 = rocdl.cvt_pk_fp8_f32(T.i32, v[2], v[3], w0, 1)
            w1 = rocdl.cvt_pk_fp8_f32(T.i32, v[4], v[5], c_zero_i32, 0)
            w1 = rocdl.cvt_pk_fp8_f32(T.i32, v[6], v[7], w1, 1)
            return _pack_i32x2(w0, w1)
        if const_expr(IS_GFX950):
            return (_pack8(p_exp_vals[0:8]), _pack8(p_exp_vals[8:16]))
        return _pack8(p_exp_vals)

    def _rescale_oaccu(oaccu, rescale):
        """Multiply all oaccu accumulators by rescale factor.
        Descending s_setprio 3->0 across 4 groups of 8 muls."""
        rv = _raw(Vec.filled(4, rescale, fx.Float32))
        result = [None] * len(oaccu)
        for group in range_constexpr(4):
            rocdl.s_setprio(3 - group)
            for j in range_constexpr(8):
                i = group * 8 + j
                result[i] = _f32(oaccu[i]) * rv
        return result

    def _process_tile_gemm1(p_lds_kv_base, kv_tile_start_i32, kv_end_i32, q_nope, q_rope, row_max_in, row_sum_e_in, is_first_iter, check_boundary, p_lds_kv_next_warp=None, row_kv_ld_next=None, kv_ld_col_base_arg=None, check_boundary_next=True, nn_resolve_start=None, nn_resolve_end=None, do_resolve_nn=None):
        """Process one KV tile: QK GEMM -> softmax -> V transpose -> pack P.

        GEMM2 (PV accumulation) is NOT included -- call _gemm2_with_rescale
        after the branch merge to keep oaccu out of phi nodes.

        Returns (row_max, row_sum_e, p_pack, rescale).
        """
        k_base_i32 = _i32(ArithValue(p_lds_kv_base) + k_lds_lane_offset)
        do_prefetch = p_lds_kv_next_warp is not None

        def _maybe_prefetch(block_idx):
            """Issue prefetch (OOB check controlled by check_boundary_next)."""
            if const_expr(not do_prefetch):
                return
            row_p0_next, row_p1_next = row_kv_ld_next
            _prefetch_k_tile_asm(p_lds_kv_next_warp, row_p0_next, kv_ld_col_base_arg, block_idx, pass_idx=0, check_boundary=check_boundary_next)
            if const_expr(IS_GFX950):
                _prefetch_k_tile_asm(p_lds_kv_next_warp, row_p1_next, kv_ld_col_base_arg, block_idx, pass_idx=1, check_boundary=check_boundary_next)
        _maybe_prefetch(0)
        P_COMP_SUBS = BLOCK_N // MFMA_N
        p_comp = [c_zero_v4f32] * P_COMP_SUBS
        for nope_pair in range_constexpr(NUM_NOPE_ITERS):
            tile_0 = nope_pair * 2
            tile_1 = nope_pair * 2 + 1
            k0 = [_load_k_from_lds(k_base_i32, 16 * h, tile_0 * BLOCK_K) for h in range_constexpr(P_COMP_SUBS)]
            k1 = [_load_k_from_lds(k_base_i32, 16 * h, tile_1 * BLOCK_K) for h in range_constexpr(P_COMP_SUBS)]
            _maybe_prefetch(nope_pair + 1)
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=P_COMP_SUBS))
            q_0 = q_nope[tile_0]
            q_1 = q_nope[tile_1]
            if const_expr(nope_pair == 0):
                for h in range_constexpr(P_COMP_SUBS):
                    p_comp[h] = _mfma_fp8(T.f32x4, [k0[h], q_0, c_zero_v4f32, 0, 0, 0])
                rocdl.s_setprio(15)
            else:
                for h in range_constexpr(P_COMP_SUBS):
                    p_comp[h] = _mfma_fp8(T.f32x4, [k0[h], q_0, p_comp[h], 0, 0, 0])
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            for h in range_constexpr(P_COMP_SUBS):
                p_comp[h] = _mfma_fp8(T.f32x4, [k1[h], q_1, p_comp[h], 0, 0, 0])
        for rope_pair in range_constexpr(NUM_ROPE_ITERS):
            tile_0 = rope_pair * 2
            tile_1 = rope_pair * 2 + 1
            k0 = [_load_k_from_lds(k_base_i32, 16 * h, (tile_0 + 16) * BLOCK_K) for h in range_constexpr(P_COMP_SUBS)]
            k1 = [_load_k_from_lds(k_base_i32, 16 * h, (tile_1 + 16) * BLOCK_K) for h in range_constexpr(P_COMP_SUBS)]
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=P_COMP_SUBS))
            for h in range_constexpr(P_COMP_SUBS):
                p_comp[h] = _mfma_fp8(T.f32x4, [k0[h], q_rope[tile_0], p_comp[h], 0, 0, 0])
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            for h in range_constexpr(P_COMP_SUBS):
                p_comp[h] = _mfma_fp8(T.f32x4, [k1[h], q_rope[tile_1], p_comp[h], 0, 0, 0])
        rocdl.s_setprio(14)
        p_vals = []
        for sub in range_constexpr(P_COMP_SUBS):
            p_comp_sub = Vec(p_comp[sub])
            for ii in range_constexpr(4):
                p_vals.append(p_comp_sub[ii])
        if const_expr(not IS_GFX950):
            v8_raw = _load_v_from_lds(p_lds_kv_base, warp_idx, lane_idx)
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            rocdl.sched_barrier(0)
        if const_expr(IS_GFX950):
            if const_expr(do_resolve_nn is not None):
                row_kv_ld_nn_p0 = fx.Int32(-1)
                row_kv_ld_nn_p1 = fx.Int32(-1)
                if do_resolve_nn:
                    row_kv_ld_nn_p0 = _get_kv_ld_row(nn_resolve_start, nn_resolve_end, True, pass_idx=0)
                    row_kv_ld_nn_p1 = _get_kv_ld_row(nn_resolve_start, nn_resolve_end, True, pass_idx=1)
            else:
                row_kv_ld_nn_p0 = fx.Int32(-1)
                row_kv_ld_nn_p1 = fx.Int32(-1)
        elif const_expr(do_resolve_nn is not None):
            row_kv_ld_nn = fx.Int32(-1)
            if do_resolve_nn:
                row_kv_ld_nn = _get_kv_ld_row(nn_resolve_start, nn_resolve_end, True)
        else:
            row_kv_ld_nn = fx.Int32(-1)
        p_exp_vals, row_max_new, row_sum_e_new, rescale = _softmax(p_vals, row_max_in, row_sum_e_in, is_first_iter, kv_tile_start_i32, kv_end_i32, check_boundary)
        p_pack = _pack_p_to_fp8(p_exp_vals)
        if const_expr(IS_GFX950):
            p_pack_lo, p_pack_hi = p_pack
            return (row_max_new, row_sum_e_new, p_pack_lo, p_pack_hi, rescale, row_kv_ld_nn_p0, row_kv_ld_nn_p1)
        vt8 = _transpose_v(v8_raw)
        vt_lds_base = lds_base_idx + P_LDS_VT
        _store_vt_to_lds(vt_lds_base, warp_idx, lane_idx, vt8)
        return (row_max_new, row_sum_e_new, p_pack, fx.Int64(0), rescale, row_kv_ld_nn, fx.Int32(-1))

    def _gemm2_core(p_pack, oaccu, vt_base_i32):
        """GEMM2 PV accumulation loop (shared by first-iter and rescale paths)."""
        K_HALVES = BLOCK_N // 32
        rocdl.s_setprio(15)
        for pv_pair in range_constexpr(NUM_PV_ITERS // 2):
            iter_a = pv_pair * 2
            iter_b = pv_pair * 2 + 1
            col_a_strip = iter_a * MFMA_N * 2
            col_b_strip = iter_b * MFMA_N * 2
            if const_expr(K_HALVES == 2):
                p_lo, p_hi = p_pack
                a_h0_top, a_h0_bot = _issue_v_strip(vt_base_i32, 0, col_a_strip)
                a_h1_top, a_h1_bot = _issue_v_strip(vt_base_i32, 32, col_a_strip)
                b_h0_top, b_h0_bot = _issue_v_strip(vt_base_i32, 0, col_b_strip)
                b_h1_top, b_h1_bot = _issue_v_strip(vt_base_i32, 32, col_b_strip)
                read_top = [a_h0_top, a_h1_top, b_h0_top, b_h1_top]
                read_bot = [a_h0_bot, a_h1_bot, b_h0_bot, b_h1_bot]
                p_args = [p_lo, p_hi, p_lo, p_hi]
                iter_idxs = [iter_a, iter_a, iter_b, iter_b]
                wait_lgkm = [6, 4, 2, 0]
            else:
                col_a0 = col_a_strip
                col_a1 = col_a0 + MFMA_N
                col_b0 = col_b_strip
                col_b1 = col_b0 + MFMA_N
                vta0_lo, vta0_hi = _load_vt_from_lds(vt_base_i32, col_a0)
                vta1_lo, vta1_hi = _load_vt_from_lds(vt_base_i32, col_a1)
                vtb0_lo, vtb0_hi = _load_vt_from_lds(vt_base_i32, col_b0)
                vtb1_lo, vtb1_hi = _load_vt_from_lds(vt_base_i32, col_b1)
                read0_lo = [vta0_lo, vtb0_lo]
                read0_hi = [vta0_hi, vtb0_hi]
                read1_lo = [vta1_lo, vtb1_lo]
                read1_hi = [vta1_hi, vtb1_hi]
                p_args = [p_pack, p_pack]
                iter_idxs = [iter_a, iter_b]
                wait_lgkm = [4, 0]
            for step in range_constexpr(K_HALVES * 2):
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=wait_lgkm[step]))
                if const_expr(K_HALVES == 2):
                    lhs0, lhs1 = _v_swap_pair(read_top[step], read_bot[step])
                else:
                    lhs0 = _pack_i32x2(read0_lo[step], read0_hi[step])
                    lhs1 = _pack_i32x2(read1_lo[step], read1_hi[step])
                iter_idx = iter_idxs[step]
                p_arg = p_args[step]
                acc_idx = iter_idx * 2
                oaccu[acc_idx] = _mfma_fp8(T.f32x4, [lhs0, p_arg, oaccu[acc_idx], 0, 0, 0])
                oaccu[acc_idx + 1] = _mfma_fp8(T.f32x4, [lhs1, p_arg, oaccu[acc_idx + 1], 0, 0, 0])
            rocdl.sched_barrier(0)
            if const_expr(pv_pair < NUM_PV_ITERS // 2 - 1):
                rocdl.s_nop(1)
        rocdl.s_setprio(0)
        return oaccu

    def _gemm2_first_iter(p_pack, vt_base_i32):
        """GEMM2 for first iteration: C=0 (hardcoded), no rescale.

        The MFMA C input is literal c_zero_v4f32, so LLVM doesn't need
        oaccu registers live -- results go to fresh registers.
        """
        _barrier(lgkmcnt=0)
        rocdl.sched_barrier(0)
        oaccu = [c_zero_v4f32] * (NUM_PV_ITERS * 2)
        return _gemm2_core(p_pack, oaccu, vt_base_i32)

    def _gemm2_with_rescale(p_pack, rescale, oaccu_in, vt_base_i32):
        """Rescale oaccu, barrier, then GEMM2 PV accumulation.

        This runs after the branch merge so oaccu never enters phi nodes.
        """
        oaccu = _rescale_oaccu(oaccu_in, rescale)
        _barrier(lgkmcnt=0)
        rocdl.sched_barrier(0)
        return _gemm2_core(p_pack, oaccu, vt_base_i32)

    def _pack_f32x4_to_bf16_2dw(acc_val):
        """Convert f32x4 accumulator to 2 packed bf16 dwords."""
        i16s = Vec(acc_val).to(fx.BFloat16).bitcast(fx.Int16)
        i16_0, i16_1, i16_2, i16_3 = (_raw(i16s[j]) for j in range(4))
        dw0 = _raw(ArithValue(i16_0).extui(T.i32) | ArithValue(i16_1).extui(T.i32) << 16)
        dw1 = _raw(ArithValue(i16_2).extui(T.i32) | ArithValue(i16_3).extui(T.i32) << 16)
        return (dw0, dw1)

    def _store_oaccu_pair_bf16(oaccu_a, oaccu_b, tile_idx, p_lds_o, row_base_i32):
        """Store 2 oaccu groups (1 PV iter) as bf16 via LDS reshape.

        Matches HK OManager16bitsV2: writes MFMA-layout data to LDS,
        reads back in row-major coalesced layout, then buffer_store_dwordx4.
        """
        o16_row_st = lane_idx % 16
        o16_col_st = lane_idx / 16 * 4
        o16_st_offset = _raw((o16_row_st / 2 * O16_ELEM_PER_PAD_2ROWS + o16_row_st % 2 * O16_NUM_COLS + o16_col_st) * 2)
        o16_row_ld = lane_idx / 4
        o16_col_ld = lane_idx % 4 * 8
        o16_rd_offset = _raw((o16_row_ld / 2 * O16_ELEM_PER_PAD_2ROWS + o16_row_ld % 2 * O16_NUM_COLS + o16_col_ld) * 2)
        lds_warp = ArithValue(p_lds_o) + warp_idx * O16_LDS_PER_WARP
        lds_st_addr = _i32(ArithValue(lds_warp) + o16_st_offset)
        for sub, acc_val in enumerate([oaccu_a, oaccu_b]):
            dw0, dw1 = _pack_f32x4_to_bf16_2dw(acc_val)
            vec_2dw = Vec.from_elements([dw0, dw1], fx.Int32)
            sub_offset = sub * O16_NUM_COLS
            st_addr_sub = _i32(ArithValue(lds_st_addr) + sub_offset)
            st_ptr = _lds_ptr_from_i32(st_addr_sub)
            _ptr_store(vec_2dw, st_ptr, alignment=8, volatile_=True)
        rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
        lds_rd_addr = _i32(ArithValue(lds_warp) + o16_rd_offset)
        rd_ptr = _lds_ptr_from_i32(lds_rd_addr)
        data = _ptr_load(T.i32x4, rd_ptr, alignment=16)
        rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
        row_vram = ArithValue(row_base_i32) + o16_row_ld
        col_vram = ArithValue(o16_col_ld) + tile_idx * MFMA_N * 2
        vram_offset = _raw((row_vram * V_HEAD_DIM + col_vram) * 2)
        buffer_ops.buffer_store(data, final_output_rsrc, vram_offset, offset_is_bytes=True)

    def _store_oaccu_pair_split(oaccu_a, oaccu_b, tile_idx, p_lds_o, row_base_i32):
        """Store 2 oaccu groups (1 PV iter) as f32 via LDS reshape.

        Matches HK OManager32bitsV2: writes MFMA-layout f32 data to LDS,
        reads back in row-major coalesced layout, then buffer_store_dwordx4.
        16 rows need 2 rounds (8 rows each) because 64 lanes / 8 lanes-per-row = 8.
        """
        o32_row_st = lane_idx % 16
        o32_col_st = lane_idx / 16 * 4
        o32_st_offset = (o32_row_st * O32_ELEM_PER_PAD_ROW + o32_col_st) * 4
        o32_row_ld = lane_idx / 8
        o32_col_ld = lane_idx % 8 * 4
        o32_rd_offset = (o32_row_ld * O32_ELEM_PER_PAD_ROW + o32_col_ld) * 4
        lds_warp = ArithValue(p_lds_o) + warp_idx * O32_LDS_PER_WARP
        lds_st_addr = _i32(ArithValue(lds_warp) + o32_st_offset)
        col_offset_i32 = tile_idx * MFMA_N * 2
        O32_LD_DELTA = 8 * O32_ELEM_PER_PAD_ROW * 4
        rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
        for sub, acc_val in enumerate([oaccu_a, oaccu_b]):
            sub_offset = sub * O32_NUM_COLS // 2 * 4
            st_addr_sub = _i32(ArithValue(lds_st_addr) + sub_offset)
            st_ptr = _lds_ptr_from_i32(st_addr_sub)
            _ptr_store(acc_val, st_ptr, alignment=16)
        rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
        lds_rd_addr = _i32(ArithValue(lds_warp) + o32_rd_offset)
        rd_ptr = _lds_ptr_from_i32(lds_rd_addr)
        data_0 = _ptr_load(T.f32x4, rd_ptr, alignment=16)
        data_1 = _ptr_load(T.f32x4, _gep(rd_ptr, static_byte_offset=O32_LD_DELTA), alignment=16)
        rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
        row_vram_0 = ArithValue(row_base_i32) + o32_row_ld
        col_vram = ArithValue(o32_col_ld) + col_offset_i32
        vram_off_0 = _raw((row_vram_0 * V_HEAD_DIM + col_vram) * 4)
        buffer_ops.buffer_store(_raw(Vec(data_0).bitcast(fx.Int32)), split_output_rsrc, vram_off_0, offset_is_bytes=True)
        row_vram_1 = row_vram_0 + 8
        vram_off_1 = _raw((row_vram_1 * V_HEAD_DIM + col_vram) * 4)
        buffer_ops.buffer_store(_raw(Vec(data_1).bitcast(fx.Int32)), split_output_rsrc, vram_off_1, offset_is_bytes=True)

    def _gemm2_last_with_store(p_pack, rescale, oaccu_in, vt_base_i32, reci_sum, is_split, p_lds_o, row_base_i32, is_first_iter_flag):
        """Last-tile GEMM2: interleave rescale + MFMA + normalize + store."""
        K_HALVES = BLOCK_N // 32
        rescale_vec = _raw(Vec.filled(4, fx.Float32(rescale), fx.Float32))
        reci_vec = _raw(Vec.filled(4, fx.Float32(reci_sum), fx.Float32))
        _barrier(lgkmcnt=0)
        rocdl.sched_barrier(0)
        rocdl.s_setprio(15)
        for pv_pair in range_constexpr(NUM_PV_ITERS // 2):
            iter_a = pv_pair * 2
            iter_b = pv_pair * 2 + 1
            col_a_strip = iter_a * MFMA_N * 2
            col_b_strip = iter_b * MFMA_N * 2
            if const_expr(not is_first_iter_flag):
                for idx in [iter_a * 2, iter_a * 2 + 1, iter_b * 2, iter_b * 2 + 1]:
                    oaccu_in[idx] = _f32(oaccu_in[idx]) * rescale_vec
            if const_expr(K_HALVES == 2):
                p_lo, p_hi = p_pack
                a_h0_top, a_h0_bot = _issue_v_strip(vt_base_i32, 0, col_a_strip)
                a_h1_top, a_h1_bot = _issue_v_strip(vt_base_i32, 32, col_a_strip)
                b_h0_top, b_h0_bot = _issue_v_strip(vt_base_i32, 0, col_b_strip)
                b_h1_top, b_h1_bot = _issue_v_strip(vt_base_i32, 32, col_b_strip)
                read_top = [a_h0_top, a_h1_top, b_h0_top, b_h1_top]
                read_bot = [a_h0_bot, a_h1_bot, b_h0_bot, b_h1_bot]
                p_args = [p_lo, p_hi, p_lo, p_hi]
                iter_idxs = [iter_a, iter_a, iter_b, iter_b]
                wait_lgkm = [6, 4, 2, 0]
            else:
                col_a0 = col_a_strip
                col_a1 = col_a0 + MFMA_N
                col_b0 = col_b_strip
                col_b1 = col_b0 + MFMA_N
                vta0_lo, vta0_hi = _load_vt_from_lds(vt_base_i32, col_a0)
                vta1_lo, vta1_hi = _load_vt_from_lds(vt_base_i32, col_a1)
                vtb0_lo, vtb0_hi = _load_vt_from_lds(vt_base_i32, col_b0)
                vtb1_lo, vtb1_hi = _load_vt_from_lds(vt_base_i32, col_b1)
                read0_lo = [vta0_lo, vtb0_lo]
                read0_hi = [vta0_hi, vtb0_hi]
                read1_lo = [vta1_lo, vtb1_lo]
                read1_hi = [vta1_hi, vtb1_hi]
                p_args = [p_pack, p_pack]
                iter_idxs = [iter_a, iter_b]
                wait_lgkm = [4, 0]
            for step in range_constexpr(K_HALVES * 2):
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=wait_lgkm[step]))
                if const_expr(K_HALVES == 2):
                    lhs0, lhs1 = _v_swap_pair(read_top[step], read_bot[step])
                else:
                    lhs0 = _pack_i32x2(read0_lo[step], read0_hi[step])
                    lhs1 = _pack_i32x2(read1_lo[step], read1_hi[step])
                iter_idx = iter_idxs[step]
                p_arg = p_args[step]
                acc_idx = iter_idx * 2
                acc0 = _mfma_fp8(T.f32x4, [lhs0, p_arg, oaccu_in[acc_idx], 0, 0, 0])
                acc1 = _mfma_fp8(T.f32x4, [lhs1, p_arg, oaccu_in[acc_idx + 1], 0, 0, 0])
                oaccu_in[acc_idx] = acc0
                oaccu_in[acc_idx + 1] = acc1
            rocdl.sched_barrier(0)
            acc_a0 = _f32(oaccu_in[iter_a * 2]) * reci_vec
            acc_a1 = _f32(oaccu_in[iter_a * 2 + 1]) * reci_vec
            acc_b0 = _f32(oaccu_in[iter_b * 2]) * reci_vec
            acc_b1 = _f32(oaccu_in[iter_b * 2 + 1]) * reci_vec
            if const_expr(is_split):
                _store_oaccu_pair_split(acc_a0, acc_a1, iter_a, p_lds_o, row_base_i32)
                _store_oaccu_pair_split(acc_b0, acc_b1, iter_b, p_lds_o, row_base_i32)
            else:
                _store_oaccu_pair_bf16(acc_a0, acc_a1, iter_a, p_lds_o, row_base_i32)
                _store_oaccu_pair_bf16(acc_b0, acc_b1, iter_b, p_lds_o, row_base_i32)
        rocdl.s_setprio(0)
    p_lds_kv_0_offset = V3_P_LDS_KV_0 if IS_GFX950 else P_LDS_KV_0
    p_lds_kv_1_offset = V3_P_LDS_KV_1 if IS_GFX950 else P_LDS_KV_1
    kv_warp_stride = V3_KV_BYTES_PER_2SUB_PADDED if IS_GFX950 else KV_SUB_BYTES
    p_lds_kv_0_base = lds_base_idx + p_lds_kv_0_offset
    p_lds_kv_1_base = lds_base_idx + p_lds_kv_1_offset

    def _kv_warp_lds_base(p_lds_kv_base):
        """Return this warp's KV LDS base as a uniform i32 address."""
        warp_offset = _raw(ArithValue(_uniform_i32(warp_idx)) * kv_warp_stride)
        return _raw(ArithValue(_i32(p_lds_kv_base)) + warp_offset)
    p_lds_kv_0_warp = _kv_warp_lds_base(p_lds_kv_0_base)
    p_lds_kv_1_warp = _kv_warp_lds_base(p_lds_kv_1_base)

    def _vt_base_i32():
        vt_row_blk = lane_idx / 16
        vt_col_blk = lane_idx % 16 / VT_COLS_PER_THR
        vt_row_inblk = lane_idx % VT_ROWS_PER_THR
        vt_col_inblk = lane_idx % 8 / VT_ROWS_PER_THR * VT_ROWS_PER_THR
        vt_block_offset = (vt_row_blk * VT_BLKS_PER_ROW_PAD + vt_col_blk) * VT_ELEMS_PER_BLK
        vt_inblock_offset = vt_row_inblk * VT_COLS_PER_THR + vt_col_inblk
        vt_lds_lane_offset = vt_block_offset + vt_inblock_offset
        return _i32(ArithValue(lds_base_idx + P_LDS_VT) + vt_lds_lane_offset)
    if const_expr(IS_GFX950):
        v_lane_in_grp = lane_idx % 16
        v_row_lane = lane_idx / 16 * 4 + v_lane_in_grp / 2 % 4
        v_col_lane = (lane_idx % 2 + v_lane_in_grp / 8 * 2) * 8
        v_lds_lane_offset = v_row_lane / V3_KV_SUB_BLOCK_ROWS * V3_KV_BYTES_PER_2SUB_PADDED + v_row_lane % V3_KV_SUB_BLOCK_ROWS * V3_KV_SUB_BLOCK_COLS + v_col_lane
        V_TR8_RES_TYPE = Vec.make_type(2, fx.Int32)

        def _load_v_tr_from_lds(v_base_i32, row_offset, col_offset):
            """gfx950 ds_read_b64_tr_b8: HW transpose-during-load, 8 fp8 per lane.

            Same fixed_offset formula as K load (V3 LDS layout); only the per-lane
            base differs (v_lds_lane_offset vs k_lds_lane_offset). Returns a
            vector<2xi32> (= 64 bits = 8 fp8 elements per lane).

            row_offset: 0/16/32/48 (which 16-row half).
            col_offset: column offset in elements (multiple of 32).
            """
            fixed_offset = row_offset // 32 * V3_KV_ROW_PASS_SLOT_STRIDE * V3_KV_BYTES_PER_2SUB_PADDED + row_offset % 32 // 16 * V3_KV_BYTES_PER_SUB_BLOCK + col_offset // KV_NUM_COLS * V3_KV_BYTES_PER_BLOCK + col_offset % KV_NUM_COLS // V3_KV_SUB_BLOCK_COLS * V3_KV_NUM_WARPS_PER_COL * V3_KV_BYTES_PER_2SUB_PADDED
            ptr = _lds_ptr_from_i32(v_base_i32, byte_offset=fixed_offset)
            return rocdl.ds_read_tr8_b64(V_TR8_RES_TYPE, ptr).result

        def _v_swap_pair(pair_top_v2i32, pair_bot_v2i32):
            """Mirror v_swap_b32 v[top+1], v[bot]: returns (b0_i64, b1_i64)."""
            top_lo = Vec(pair_top_v2i32)[0]
            top_hi = Vec(pair_top_v2i32)[1]
            bot_lo = Vec(pair_bot_v2i32)[0]
            bot_hi = Vec(pair_bot_v2i32)[1]
            b0_i64 = _pack_i32x2(top_lo, bot_lo)
            b1_i64 = _pack_i32x2(top_hi, bot_hi)
            return (b0_i64, b1_i64)

        def _issue_v_strip(v_base_i32, k_half_row_base, col_strip):
            """Issue 2 ds_read_b64_tr_b8 (NO swap). Returns the 2 raw v2i32 load
            results. Caller MUST place an `s_waitcnt lgkmcnt(N)` that drains these
            loads BEFORE calling `_v_swap_pair` -- the swap is an SSA dword
            extract/repack that LLVM can otherwise hoist above the waitcnt and
            consume stale register values.
            """
            pair_top = _load_v_tr_from_lds(v_base_i32, k_half_row_base + 0, col_strip)
            pair_bot = _load_v_tr_from_lds(v_base_i32, k_half_row_base + 16, col_strip)
            return (pair_top, pair_bot)

        def _v_base_i32(p_lds_kv_base):
            """V3: V is read transposed-during-load directly from the KV LDS region
            of the current double buffer. Per-lane base = kv_base + v_lds_lane_offset.
            """
            return _i32(ArithValue(p_lds_kv_base) + v_lds_lane_offset)
    for work_idx in range(work_start_idx, work_end_idx):
        wi_base = work_idx * SIZE_MLA_WORK_INFO_IN_DW
        wi_dw1_4 = buffer_ops.buffer_load(work_info_set_rsrc, wi_base + 1, vec_width=4, dtype=T.i32)
        wi_dw5_6 = buffer_ops.buffer_load(work_info_set_rsrc, wi_base + 5, vec_width=2, dtype=T.i32)
        wi_dw1_4_vec = Vec(wi_dw1_4)
        wi_dw5_6_vec = Vec(wi_dw5_6)
        partial_qo_loc = rocdl.readfirstlane(T.i32, wi_dw1_4_vec[0])
        qo_start = rocdl.readfirstlane(T.i32, wi_dw1_4_vec[1])
        qo_end = rocdl.readfirstlane(T.i32, wi_dw1_4_vec[2])
        kv_start = rocdl.readfirstlane(T.i32, wi_dw1_4_vec[3])
        kv_end = rocdl.readfirstlane(T.i32, wi_dw5_6_vec[0])
        kv_offset = rocdl.readfirstlane(T.i32, wi_dw5_6_vec[1])
        query_span = qo_end - qo_start
        valid_query_waves = query_span * waves_per_query
        query_wave_idx = arith.minsi(_i32(warp_idx), _i32(valid_query_waves - 1))
        query_position = query_wave_idx // waves_per_query
        query_position_from_last = query_span - 1 - query_position
        causal_offset = arith.maxsi(
            _i32(query_position_from_last - kv_offset), _raw(fx.Int32(0))
        )
        kv_end_eff = kv_end - causal_offset
        kv_len = kv_end - kv_start
        row_max = c_neg_inf
        row_sum_e = c_zero_f32
        kv_len_v = ArithValue(kv_len)
        num_tiles = (kv_len_v + BLOCK_N - 1).with_signedness(False) // BLOCK_N
        first_tile_needs_boundary = kv_len_v < BLOCK_N
        has_multi_tiles = kv_len_v > BLOCK_N
        last_tile_partial = kv_len_v & BLOCK_N - 1 != 0
        row_kv_ld_first_p0 = fx.Int32(-1)
        row_kv_ld_first_p1 = fx.Int32(-1)
        if first_tile_needs_boundary:
            row_kv_ld_first_p0 = _get_kv_ld_row(kv_start, kv_end, True, pass_idx=0)
            if const_expr(IS_GFX950):
                row_kv_ld_first_p1 = _get_kv_ld_row(kv_start, kv_end, True, pass_idx=1)
        else:
            kv_first_end = _raw(ArithValue(kv_start) + BLOCK_N)
            row_kv_ld_first_p0 = _get_kv_ld_row(kv_start, kv_first_end, False, pass_idx=0)
            if const_expr(IS_GFX950):
                row_kv_ld_first_p1 = _get_kv_ld_row(kv_start, kv_first_end, False, pass_idx=1)
        q_nope_packs, q_rope_packs = _load_q_to_regs(qo_start, query_wave_idx)
        if first_tile_needs_boundary:
            _async_load_kv_all(p_lds_kv_0_warp, row_kv_ld_first_p0, kv_ld_col_base, row_p1_i32=row_kv_ld_first_p1, check_boundary=True)
        else:
            _async_load_kv_all(p_lds_kv_0_warp, row_kv_ld_first_p0, kv_ld_col_base, row_p1_i32=row_kv_ld_first_p1, check_boundary=False)
        kv_start_v = ArithValue(kv_start)
        kv_start_plus_bn = _raw(kv_start_v + BLOCK_N)
        kv_start_plus_2bn = _raw(kv_start_v + 2 * BLOCK_N)
        tile1_is_full = ArithValue(kv_start_plus_2bn) <= kv_end
        row_kv_ld_tile1_p0 = fx.Int32(-1)
        row_kv_ld_tile1_p1 = fx.Int32(-1)
        if tile1_is_full:
            row_kv_ld_tile1_p0 = _get_kv_ld_row(kv_start_plus_bn, kv_start_plus_2bn, False, pass_idx=0)
            if const_expr(IS_GFX950):
                row_kv_ld_tile1_p1 = _get_kv_ld_row(kv_start_plus_bn, kv_start_plus_2bn, False, pass_idx=1)
        else:
            row_kv_ld_tile1_p0 = _get_kv_ld_row(kv_start_plus_bn, _raw(kv_end), True, pass_idx=0)
            if const_expr(IS_GFX950):
                row_kv_ld_tile1_p1 = _get_kv_ld_row(kv_start_plus_bn, _raw(kv_end), True, pass_idx=1)
        row_kv_ld_tile1_arg = (row_kv_ld_tile1_p0, row_kv_ld_tile1_p1)
        first_tile_cbn = ArithValue(kv_start_plus_2bn) > kv_end
        do_resolve_nn_first = ArithValue(kv_start_plus_2bn) < kv_end
        p_pack_first_lo = fx.Int64(0)
        p_pack_first_hi = fx.Int64(0)
        row_kv_ld_nn_first_p0 = fx.Int32(-1)
        row_kv_ld_nn_first_p1 = fx.Int32(-1)
        rescale_first = c_one_f32
        if _raw(has_multi_tiles):
            if first_tile_cbn:
                _barrier(vmcnt=0, lgkmcnt=0)
                rocdl.sched_barrier(0)
                row_max, row_sum_e, p_pack_first_lo, p_pack_first_hi, rescale_first, row_kv_ld_nn_first_p0, row_kv_ld_nn_first_p1 = _process_tile_gemm1(p_lds_kv_0_base, kv_start, kv_end_eff, q_nope_packs, q_rope_packs, row_max, row_sum_e, is_first_iter=True, check_boundary=False, p_lds_kv_next_warp=p_lds_kv_1_warp, row_kv_ld_next=row_kv_ld_tile1_arg, kv_ld_col_base_arg=kv_ld_col_base, check_boundary_next=True, nn_resolve_start=kv_start_plus_2bn, nn_resolve_end=kv_end, do_resolve_nn=do_resolve_nn_first)
            else:
                _barrier(vmcnt=0, lgkmcnt=0)
                rocdl.sched_barrier(0)
                row_max, row_sum_e, p_pack_first_lo, p_pack_first_hi, rescale_first, row_kv_ld_nn_first_p0, row_kv_ld_nn_first_p1 = _process_tile_gemm1(p_lds_kv_0_base, kv_start, kv_end_eff, q_nope_packs, q_rope_packs, row_max, row_sum_e, is_first_iter=True, check_boundary=False, p_lds_kv_next_warp=p_lds_kv_1_warp, row_kv_ld_next=row_kv_ld_tile1_arg, kv_ld_col_base_arg=kv_ld_col_base, check_boundary_next=False, nn_resolve_start=kv_start_plus_2bn, nn_resolve_end=kv_end, do_resolve_nn=do_resolve_nn_first)
        else:
            _barrier(vmcnt=0, lgkmcnt=0)
            rocdl.sched_barrier(0)
            row_max, row_sum_e, p_pack_first_lo, p_pack_first_hi, rescale_first, row_kv_ld_nn_first_p0, row_kv_ld_nn_first_p1 = _process_tile_gemm1(p_lds_kv_0_base, kv_start, kv_end_eff, q_nope_packs, q_rope_packs, row_max, row_sum_e, is_first_iter=True, check_boundary=first_tile_needs_boundary)
        if const_expr(IS_GFX950):
            p_pack_first = (p_pack_first_lo, p_pack_first_hi)
        else:
            p_pack_first = p_pack_first_lo

        def _write_lse(pqo_loc_i32, rm, rse):
            """Write LSE for split output (first 16 lanes per warp)."""
            if ArithValue(lane_idx) < 16:
                log2_sum = fmath.log2(rse, fastmath=fm_fast)
                lse = fmath.fma(log2_sum, c_inv_log2e, rm, fastmath=fm_fast)
                row_idx = _raw(
                    ArithValue(lane_idx)
                    + _idx(query_wave_idx) * 16
                    + _idx(pqo_loc_i32) * num_qheads
                )
                buffer_ops.buffer_store(lse, split_lse_rsrc, row_idx)
        p_lds_o = p_lds_kv_0_base

        def _do_last_gemm2_and_store(pp, rs, oaccu_list, rm, rse, is_first_iter_flag, v_kv_base=None, o_kv_base=None):
            """GEMM2 last tile with interleaved store + LSE write.

            Captures `partial_qo_loc` and `p_lds_o` from the enclosing work-loop
            iteration. AITER fast-mode metadata for this kernel writes partial
            split outputs for every work item; the host reduce kernel produces
            the final bf16 output.
            """
            reci = rocdl.rcp(T.f32, rse)
            rb_split = _raw(
                _idx(partial_qo_loc) * num_qheads + _idx(query_wave_idx) * 16
            )
            _write_lse(_raw(partial_qo_loc), rm, rse)
            if const_expr(BLOCK_N // 32 == 2):
                v_base = _v_base_i32(v_kv_base)
                o_base = o_kv_base
            else:
                v_base = _vt_base_i32()
                o_base = p_lds_o
            _gemm2_last_with_store(pp, rs, list(oaccu_list), v_base, reci, True, o_base, rb_split, is_first_iter_flag)

        def _multi_tile_path():
            if const_expr(IS_GFX950):
                oaccu_mt = _gemm2_first_iter(p_pack_first, _v_base_i32(p_lds_kv_0_base))
            else:
                oaccu_mt = _gemm2_first_iter(p_pack_first, _vt_base_i32())
            num_tiles_v = ArithValue(num_tiles)
            num_tiles_m1 = _raw(num_tiles_v - 1)
            num_tiles_m2 = _raw(num_tiles_v - 2)
            init_args = [row_max, row_sum_e] + oaccu_mt + [row_kv_ld_nn_first_p0, row_kv_ld_nn_first_p1]
            for tile_iv, state in range(_idx(1), _idx(num_tiles_m1), _idx(1), init=init_args):
                tile_iv_i32 = ArithValue(fx.Int32(tile_iv))
                kv_tile_start_i32 = _raw(kv_start_v + tile_iv_i32 * BLOCK_N)
                rm_carried = state[0]
                rse_carried = state[1]
                oaccu_carried = [state[2 + i] for i in range(NUM_PV_ITERS * 2)]
                row_kv_ld_next_arg = (state[2 + NUM_PV_ITERS * 2], state[2 + NUM_PV_ITERS * 2 + 1])
                is_odd = tile_iv_i32 & 1 != 0
                curr_base_idx = ArithValue(is_odd).select(p_lds_kv_1_base, p_lds_kv_0_base)
                next_warp = ArithValue(is_odd).select(p_lds_kv_0_warp, p_lds_kv_1_warp)
                is_second_to_last = tile_iv_i32 == ArithValue(num_tiles_m2)
                mid_cbn = _raw(ArithValue(is_second_to_last) & last_tile_partial)
                nn_start_mid = _raw(ArithValue(kv_tile_start_i32) + 2 * BLOCK_N)
                do_resolve_nn_mid = ArithValue(nn_start_mid) < kv_end
                rm_m = c_neg_inf
                rse_m = c_zero_f32
                rs_m = c_one_f32
                pp_m_lo = fx.Int64(0)
                pp_m_hi = fx.Int64(0)
                nn_m_p0 = fx.Int32(-1)
                nn_m_p1 = fx.Int32(-1)
                if mid_cbn:
                    _barrier(vmcnt=0, lgkmcnt=0)
                    rocdl.sched_barrier(0)
                    rm_m, rse_m, pp_m_lo, pp_m_hi, rs_m, nn_m_p0, nn_m_p1 = _process_tile_gemm1(curr_base_idx, kv_tile_start_i32, kv_end_eff, q_nope_packs, q_rope_packs, rm_carried, rse_carried, is_first_iter=False, check_boundary=False, p_lds_kv_next_warp=next_warp, row_kv_ld_next=row_kv_ld_next_arg, kv_ld_col_base_arg=kv_ld_col_base, check_boundary_next=True, nn_resolve_start=nn_start_mid, nn_resolve_end=kv_end, do_resolve_nn=do_resolve_nn_mid)
                else:
                    _barrier(vmcnt=0, lgkmcnt=0)
                    rocdl.sched_barrier(0)
                    rm_m, rse_m, pp_m_lo, pp_m_hi, rs_m, nn_m_p0, nn_m_p1 = _process_tile_gemm1(curr_base_idx, kv_tile_start_i32, kv_end_eff, q_nope_packs, q_rope_packs, rm_carried, rse_carried, is_first_iter=False, check_boundary=False, p_lds_kv_next_warp=next_warp, row_kv_ld_next=row_kv_ld_next_arg, kv_ld_col_base_arg=kv_ld_col_base, check_boundary_next=False, nn_resolve_start=nn_start_mid, nn_resolve_end=kv_end, do_resolve_nn=do_resolve_nn_mid)
                if const_expr(IS_GFX950):
                    pp_m = (pp_m_lo, pp_m_hi)
                    oa_m = _gemm2_with_rescale(pp_m, rs_m, oaccu_carried, _v_base_i32(curr_base_idx))
                else:
                    oa_m = _gemm2_with_rescale(pp_m_lo, rs_m, oaccu_carried, _vt_base_i32())
                yield_vals = [rm_m, rse_m] + oa_m + [nn_m_p0, nn_m_p1]
                results = (yield yield_vals)
            row_max_mt = results[0]
            row_sum_e_mt = results[1]
            oaccu_mt = [results[2 + i] for i in range(NUM_PV_ITERS * 2)]
            last_tile_iv = ArithValue(num_tiles_m1)
            kv_last_start = _raw(kv_start_v + last_tile_iv * BLOCK_N)
            last_is_odd = last_tile_iv & 1 != 0
            last_curr_base = ArithValue(last_is_odd).select(p_lds_kv_1_base, p_lds_kv_0_base)
            last_o_base = ArithValue(last_is_odd).select(p_lds_kv_0_base, p_lds_kv_1_base)
            _barrier(vmcnt=0, lgkmcnt=0)
            rocdl.sched_barrier(0)
            rm_l, rse_l, pp_l_lo, pp_l_hi, rs_l, _nn_l_p0, _nn_l_p1 = _process_tile_gemm1(last_curr_base, kv_last_start, kv_end_eff, q_nope_packs, q_rope_packs, row_max_mt, row_sum_e_mt, is_first_iter=False, check_boundary=last_tile_partial)
            if const_expr(IS_GFX950):
                pp_l = (pp_l_lo, pp_l_hi)
            else:
                pp_l = pp_l_lo
            _do_last_gemm2_and_store(pp_l, rs_l, oaccu_mt, rm_l, rse_l, is_first_iter_flag=False, v_kv_base=last_curr_base, o_kv_base=last_o_base)

        def _single_tile_path():
            oaccu_st = [c_zero_v4f32] * (NUM_PV_ITERS * 2)
            _do_last_gemm2_and_store(p_pack_first, rescale_first, oaccu_st, row_max, row_sum_e, is_first_iter_flag=True, v_kv_base=p_lds_kv_0_base, o_kv_base=p_lds_kv_1_base)

        @flyc.jit
        def _dispatch_multi_single():
            if has_multi_tiles:
                _multi_tile_path()
            else:
                _single_tile_path()
        _dispatch_multi_single()

_MLA_COMPILED_LOCK = threading.Lock()
_MLA_COMPILED = {}


@flyc.jit
def launch_mla_fwd_decode_m16x8_fp8_fp8(
    query: fx.Tensor,
    kv_buffer: fx.Tensor,
    kv_page_indices: fx.Tensor,
    work_indptr: fx.Tensor,
    work_info_set: fx.Tensor,
    final_output: fx.Tensor,
    split_output: fx.Tensor,
    split_lse: fx.Tensor,
    q_scale: fx.Tensor,
    kv_scale: fx.Tensor,
    softmax_scale: fx.Float32,
    num_qheads: fx.Constexpr[int],
    num_cus: fx.Constexpr,
    lds_size: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    assert TOTAL_LDS_BYTES <= lds_size
    kn_mla_fwd_decode_m16x8_fp8_fp8(
        query,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info_set,
        final_output,
        split_output,
        split_lse,
        q_scale,
        kv_scale,
        softmax_scale,
        num_qheads,
    ).launch(
        grid=(num_cus, 1, 1),
        block=(NUM_THREADS, 1, 1),
        smem=0,
        stream=stream,
    )


def run_mla_fwd_decode_m16x8_fp8_fp8(
    query,
    kv_buffer,
    kv_page_indices,
    work_indptr,
    work_info_set,
    final_output,
    split_output,
    split_lse,
    q_scale,
    kv_scale,
    softmax_scale,
    num_qheads,
    num_cus,
    lds_size,
    stream,
):
    args = (
        query,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info_set,
        final_output,
        split_output,
        split_lse,
        q_scale,
        kv_scale,
        softmax_scale,
        num_qheads,
        num_cus,
        lds_size,
        stream,
    )
    key = (
        query.device.index,
        query.dtype,
        kv_buffer.dtype,
        num_qheads,
        num_cus,
        lds_size,
    )
    compiled = _MLA_COMPILED.get(key)
    if compiled is None:
        with _MLA_COMPILED_LOCK:
            compiled = _MLA_COMPILED.get(key)
            if compiled is None:
                compiled = flyc.compile(launch_mla_fwd_decode_m16x8_fp8_fp8, *args)
                _MLA_COMPILED[key] = compiled
                return
    compiled(*args)
