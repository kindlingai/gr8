//! The fabric plan (fabric_plan.h, docs/HOST-CORE.md R6): every g6::Fabric g6m4 creates, in creation order, as a
//! function of the options, the knobs the bring-up reads, the build (model, width, rows, the constants the sizing
//! rules read) and a few values the C++ computes with helpers its kernels share. Each entry is a whole FabricConfig
//! without its address lists: a fabric's config starts as a copy of the main one as the bring-up has changed it so far.
//! fabric_plan.h rebuilds its C++ FabricPlan from the result; s2/tests/fabric_plan_vectors_test.cpp holds the result
//! to transcripts of the C++ plan it replaced, and the refusals keep their text.

use std::ffi::{c_char, CString};

use crate::cp_rows;
use crate::cstd::{c_strtol10, cs};
use crate::module_info::{MODEL_DS41, MODEL_GLM53F};
use crate::plan_sizes::{self as ps, sz, text, CpColls, Str};

/// fabric_plan.h FabKind, in main()'s creation order
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Kind {
    Main,
    LaneB,
    Prefill,
    Prefill2,
    PoolIds,
    PoolIds2,
    RouteSplit,
    RouteSplit2,
    Draft,
    DSpark,
    Dflash,
    QpWire,
    Cp,
    Gather,
    CpStage,
    DraftGather,
}
pub const KINDS: usize = 16;

/// Each fabric's port: the options' port + this. A fabric's TCP bootstrap takes four ports from there. Flash's pool-ID
/// fabrics and GLM's gathers share +64 and +72: no build creates both (ports_distinct_per_model).
pub const PORT_OFF: [i32; KINDS] = [0, 96, 8, 24, 64, 72, 80, 88, 16, 16, 40, 56, 32, 64, 48, 72];

/// The proxy core: N - off, or unpinned below it (the options' core is N)
fn cpu(proxy: i32, off: i32) -> i32 {
    if proxy > off - 1 {
        proxy - off
    } else {
        -1
    }
}

/// g6plan_cfg: a g6::FabricConfig without ips, ips2 and ring_devs
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Cfg {
    pub rank: i32,
    pub stripe_min: usize,
    pub slot_bytes: usize,
    pub run1: i32,
    pub run2: i32,
    pub slot_bytes1: usize,
    pub slot_bytes2: usize,
    pub run3: i32,
    pub slot_bytes3: usize,
    pub pool: i32,
    pub sge2: bool,
    pub compact: bool,
    pub ncoll: i32,
    pub port: i32,
    pub proxy_cpu: i32,
    pub vcap: *const u32,
    pub n_vcap: usize,
    pub vdst_k: *const u8,
    pub n_vdst_k: usize,
    pub bulk_bytes: usize,
    pub bulk_flags: i32,
    pub xtrace: i32,
    pub ring: bool,
    pub relay_cpu: i32,
}

/// g6plan_entry: one fabric. use_pf_ips2: its rail-1 addresses are --pf-ips2's, not the main config's.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub kind: u32,
    pub late: bool,
    pub use_pf_ips2: bool,
    pub port_off: i32,
    pub cfg: Cfg,
}

/// g6plan_knobs: fabric_plan.h PlanKnobs. Tri-states are -1 unset, 0, 1; a null text is unset.
#[repr(C)]
pub struct Knobs {
    pub xtrace_pf: bool,
    pub pf_stripe: i32,
    pub pf_stripe_min: i64,
    pub pf_rs: bool,
    pub fpf_attn_p1: bool,
    pub fpf_dsa_p1: bool,
    pub fpf_p1: bool,
    pub has_qa_split_cpu: bool,
    pub qa_split_cpu: i64,
    pub cp: i32,
    pub cp_r: i64,
    pub cp_step_rows: i64,
    pub cp_s3: bool,
    pub has_cp_mmax: bool,
    pub cp_mmax: i64,
    pub has_cp_r2: bool,
    pub cp_r2: i64,
    pub ds_prefill_w: *const c_char,
    pub ds_pf_tail: *const c_char,
    pub ds_pf_pool: *const c_char,
    pub ds_pf_lanes: *const c_char,
    pub ds_pf_l20_tail: bool,
    pub ds_pf_rs: bool,
    pub ds_mhc_row_shard: bool,
    pub ds_mhc_fuse: bool,
    pub ds_mhc_y_mxfp8: bool,
    pub ds_pf_direct: bool,
    pub ds_subdump: bool,
    pub ds_stripe: bool,
    pub ds_xtrace: bool,
    pub ds_stripe_min: i64,
}

/// g6plan_build: the build, and the constants of its headers the plan reads
#[repr(C)]
pub struct Build {
    pub model: u32,
    pub nr: i32,
    pub mx: i32,
    pub hid: i32,
    pub ds_window: i32,
    pub ds_tail_rows: i32,
    pub ds_cand_source: i32,
    pub ds_late_start: i32,
    pub ds_pack_rows: i32,
    pub ds_draft_ncoll: i32,
    pub coll_bits: i32,
    pub cp: ps::CpDims,
    pub qw_half: usize,
    pub route_max_nr: i32,
}

/// g6plan_in: the build, the options the plan reads (the main config as the parse left it), the knobs, and the values
/// main computes for it: pool_payload = flash_pool_owner::payload(MP), route_payload = flash_route_split::payload(MP,
/// the --ips count), dsp_coll_max = the largest DSpark draft collective.
#[repr(C)]
pub struct In {
    pub b: Build,
    pub cfg: Cfg,
    pub n_ips: usize,
    pub n_ips2: usize,
    pub n_pf_ips2: usize,
    pub nc: i32,
    pub nx: i32,
    pub nl: i32,
    pub mp: i32,
    pub mtp: i32,
    pub lanes: i32,
    pub drafts: i32,
    pub chunk: i32,
    pub s2sess: i32,
    pub rt_win: i32,
    pub route_split_mode: i32,
    pub c8_pack_env: i32,
    pub ds_draft_cap: i32,
    pub qp_roll_s: i32,
    pub qp_wire_e: i32,
    pub s2mode: bool,
    pub dfl: bool,
    pub dsk_on: bool,
    pub layers: *const i32,
    pub n_layers: usize,
    pub row_counts: *const i32,
    pub n_row_counts: usize,
    pub knobs: Knobs,
    pub pf_sp: bool,
    pub pool_owner_mode: i32,
    pub dfl_ncoll: i32,
    pub head_payload: usize,
    pub trunk_bytes: usize,
    pub full_index: *const bool,
    pub n_full_index: usize,
    pub cp_build: bool,
    pub gather: bool,
    pub draft_gather: bool,
    pub pool_payload: usize,
    pub route_payload: usize,
    pub dsp_coll_max: usize,
}

/// g6plan_cp: cplayout.h CpColls
#[repr(C)]
pub struct CCp {
    pub nr: i32,
    pub nl: i32,
    pub mmax: i32,
    pub q_row: u32,
    pub cand_row: u32,
    pub part_row: u32,
    pub q: *const i32,
    pub n_q: usize,
    pub cand: *const i32,
    pub n_cand: usize,
    pub part: *const i32,
    pub n_part: usize,
    pub cap: *const u32,
    pub dst_k: *const u8,
    pub ncoll: usize,
}

/// g6plan_plan: fabric_plan.h FabricPlan. After a refusal n is 0; the log and the scalars hold what was planned before.
#[repr(C)]
pub struct CPlan {
    pub refusal: *const c_char,
    pub log: *const *const c_char,
    pub n_log: usize,
    pub n: i32,
    pub entries: [Entry; KINDS],
    pub dec_slot_bytes: usize,
    pub pf_rows: i32,
    pub tail: i32,
    pub i_late: i32,
    pub l20_tail: bool,
    pub rs_k: i32,
    pub quarter: i32,
    pub slice_stride: usize,
    pub rowshard: bool,
    pub y_mxfp8: bool,
    pub pf_direct: bool,
    pub pf_lanes_fab: i32,
    pub two: bool,
    pub pool_owner: bool,
    pub route_split: bool,
    pub cp_on: bool,
    pub cp_r: i32,
    pub cp_r2: i32,
    pub cp_s3: i32,
    pub cp_mmax: i32,
    pub cp_step: i32,
    pub cp_step_r: i32,
    pub cp: CCp,
}

// ---------------------------------------------------------------------------------------------------- the plan

#[derive(Clone, Debug, Default)]
struct Fc {
    rank: i32,
    stripe_min: usize,
    slot_bytes: usize,
    run1: i32,
    run2: i32,
    slot_bytes1: usize,
    slot_bytes2: usize,
    run3: i32,
    slot_bytes3: usize,
    pool: i32,
    sge2: bool,
    compact: bool,
    ncoll: i32,
    port: i32,
    proxy_cpu: i32,
    vcap: Vec<u32>,
    vdst_k: Vec<u8>,
    bulk_bytes: usize,
    bulk_flags: i32,
    xtrace: i32,
    ring: bool,
    relay_cpu: i32,
    pf_ips2: bool,
}

struct Planned {
    kind: Kind,
    late: bool,
    cfg: Fc,
}

#[derive(Default)]
struct Plan {
    log: Vec<Str>,
    fabrics: Vec<Planned>,
    dec_slot_bytes: usize,
    pf_rows: i32,
    tail: i32,
    i_late: i32,
    l20_tail: bool,
    rs_k: i32,
    quarter: i32,
    slice_stride: usize,
    rowshard: bool,
    y_mxfp8: bool,
    pf_direct: bool,
    pf_lanes_fab: i32,
    two: bool,
    pool_owner: bool,
    route_split: bool,
    cp_on: bool,
    cp_r: i32,
    cp_r2: i32,
    cp_s3: i32,
    cp_mmax: i32,
    cp_step: i32,
    cp_step_r: i32,
    cp: CpColls,
}

impl Plan {
    fn add(&mut self, kind: Kind, cfg: &Fc, late: bool) {
        self.fabrics.push(Planned { kind, late, cfg: cfg.clone() });
    }
}

fn refuse<T>(m: &str) -> Result<T, Str> {
    Err(m.as_bytes().to_vec())
}

unsafe fn slice<'a, T>(p: *const T, n: usize) -> &'a [T] {
    if n == 0 || p.is_null() {
        &[]
    } else {
        std::slice::from_raw_parts(p, n)
    }
}

/// # Safety
/// Every pointer of `i` is valid for its count, and every text null or a C string.
unsafe fn plan(i: &In, p: &mut Plan) -> Result<(), Str> {
    let k = &i.knobs;
    let b = &i.b;
    let (mx, hid, nr) = (b.mx, b.hid, b.nr);
    let (nc, nx, mp, mtp) = (i.nc, i.nx, i.mp, i.mtp);
    let (flash, ds41) = (b.model == MODEL_GLM53F, b.model == MODEL_DS41);
    let layers = slice(i.layers, i.n_layers);
    let mut cfg = Fc {
        rank: i.cfg.rank,
        stripe_min: i.cfg.stripe_min,
        slot_bytes: i.cfg.slot_bytes,
        run1: i.cfg.run1,
        run2: i.cfg.run2,
        slot_bytes1: i.cfg.slot_bytes1,
        slot_bytes2: i.cfg.slot_bytes2,
        run3: i.cfg.run3,
        slot_bytes3: i.cfg.slot_bytes3,
        pool: i.cfg.pool,
        sge2: i.cfg.sge2,
        compact: i.cfg.compact,
        ncoll: i.cfg.ncoll,
        port: i.cfg.port,
        proxy_cpu: i.cfg.proxy_cpu,
        vcap: slice(i.cfg.vcap, i.cfg.n_vcap).to_vec(),
        vdst_k: slice(i.cfg.vdst_k, i.cfg.n_vdst_k).to_vec(),
        bulk_bytes: i.cfg.bulk_bytes,
        bulk_flags: i.cfg.bulk_flags,
        xtrace: i.cfg.xtrace,
        ring: i.cfg.ring,
        relay_cpu: i.cfg.relay_cpu,
        pf_ips2: false,
    };
    // Engram: [rows][6144] BF16 slots
    cfg.slot_bytes = if nx != 0 { sz(mx).wrapping_mul(6144 * 2) } else { sz(mx).wrapping_mul(sz(hid)).wrapping_mul(2) };
    p.dec_slot_bytes = cfg.slot_bytes;
    let rank = cfg.rank;
    let mut cm = ps::Runs::default(); // the main fabric's own runs (DS41 serve); run1 = 0 = uniform
    let mut cm_pool = 0;
    let mut cmb = ps::Runs::default(); // GR8_DS_PF_LANES=2: lane B's
    if ds41 {
        if i.s2mode {
            let pf_rows = ps::ds41_prefill_rowcap(text(k.ds_prefill_w), mx);
            if pf_rows == 0 {
                return refuse("DS41 serve: GR8_DS_PREFILL_W must be MX..4096");
            }
            p.pf_rows = pf_rows;
            cfg.slot_bytes = ps::ds41_prefill_payload(pf_rows, hid, if nx != 0 { 6144 } else { 0 });
            let tail = ps::ds41_parse_tail(text(k.ds_pf_tail), b.ds_tail_rows, b.ds_window);
            if tail < 0 {
                // the boundary as it is before the boot sets l20_tail()
                return Err(ps::ds41_tail_error(text(k.ds_pf_tail), b.ds_window, b.ds_late_start));
            }
            p.tail = tail;
            p.l20_tail = k.ds_pf_l20_tail;
            if p.l20_tail && tail == 0 {
                return refuse("GR8_DS_PF_L20_TAIL=1 needs GR8_DS_PF_TAIL (layer 20 joins the bounded replay)");
            }
            let late_layer = if p.l20_tail { b.ds_cand_source } else { b.ds_late_start };
            let i_late = if tail != 0 { ps::ds41_late_start(layers, late_layer) } else { -1 };
            p.i_late = i_late;
            let mut pf_rs_on = false;
            if k.ds_pf_rs {
                if i_late < 0 {
                    return refuse("GR8_DS_PF_RS=1 needs GR8_DS_PF_TAIL (the RS collectives are the layer exchanges before the bounded-replay split)");
                }
                if pf_rows <= mx || pf_rows < nr {
                    return refuse("GR8_DS_PF_RS=1 needs GR8_DS_PREFILL_W > MX");
                }
                pf_rs_on = true;
                if nc + 2 * i_late > b.coll_bits {
                    return refuse("GR8_DS_PF_RS=1: more than 128 collectives (NC + 2 i_late)");
                }
            }
            if k.ds_mhc_row_shard {
                if !ps::ds41_rowshard_ok(pf_rows, tail, nr, pf_rs_on, k.ds_mhc_fuse) {
                    return refuse("GR8_DS_MHC_ROW_SHARD=1 needs W=4096, tail=128, TP4, PF_RS=1, MHC_FUSE=1");
                }
                if nc + 2 * i_late + 1 > b.coll_bits {
                    return refuse("GR8_DS_MHC_ROW_SHARD: more than 128 collectives");
                }
                let state = ps::ds41_state_bytes(tail, hid, i.trunk_bytes);
                if state > 1024usize.wrapping_mul(sz(hid)).wrapping_mul(2) {
                    return refuse("GR8_DS_MHC_ROW_SHARD: late state exceeds an AG slot");
                }
                p.rowshard = true;
                p.log.push(
                    format!("# rank {rank}: GR8_DS_MHC_ROW_SHARD=1 Q=1024 late=128 state_bytes={state}\n").into_bytes(),
                );
            }
            if k.ds_mhc_y_mxfp8 {
                if !p.rowshard {
                    return refuse("GR8_DS_MHC_Y_MXFP8=1 needs GR8_DS_MHC_ROW_SHARD=1");
                }
                p.y_mxfp8 = true;
                p.log.push(format!("# rank {rank}: GR8_DS_MHC_Y_MXFP8=1 lossy y all-gather\n").into_bytes());
            }
            cm = ps::ds41_fabric_runs(
                pf_rows,
                tail,
                i_late,
                i.nl,
                nx,
                mx,
                hid,
                nr,
                i.head_payload,
                pf_rs_on,
                i.c8_pack_env == 1,
                p.rowshard,
                b.ds_pack_rows,
            );
            p.rs_k = cm.rs_k;
            p.quarter = cm.quarter;
            p.slice_stride = cm.slice_stride;
            // GR8_DS_PF_POOL=P: run 0 and run 3 keep P slots each
            if let Some(pv) = text(k.ds_pf_pool).filter(|v| !v.is_empty() && *v != b"0") {
                let (pl, at_end) = c_strtol10(pv);
                if !at_end || pl < 2 || pl > i64::from(cm.run1) {
                    return refuse("GR8_DS_PF_POOL must be 0 (off) or 2..the run 0 collectives (2 i_late with GR8_DS_PF_TAIL, else 2 NL)");
                }
                cm_pool = pl as i32;
                p.log.push(
                    format!(
                        "# rank {rank} DS41 fabric slot pool: GR8_DS_PF_POOL={cm_pool} (run 0: {} collectives, {})\n",
                        cm.run1,
                        if p.rs_k != 0 { "run 3 AG pooled too" } else { "no run 3" }
                    )
                    .into_bytes(),
                );
            }
            // GR8_DS_PF_LANES=2: lane B's slot table is the main one without the C8-pack late slots, with the same pool
            p.pf_lanes_fab = ps::ds41_lanes(text(k.ds_pf_lanes));
            if p.pf_lanes_fab == 0 {
                return Err(ps::DS41_LANES_ERROR.to_bytes().to_vec());
            }
            if p.pf_lanes_fab == 2 {
                let rb = ps::ds41_fabric_runs(
                    pf_rows,
                    tail,
                    i_late,
                    i.nl,
                    nx,
                    mx,
                    hid,
                    nr,
                    i.head_payload,
                    pf_rs_on,
                    false,
                    p.rowshard,
                    b.ds_pack_rows,
                );
                if rb.rs_k != p.rs_k || rb.run1 != cm.run1 || rb.ncoll != cm.ncoll {
                    return refuse(
                        "DS41 lane B fabric: its runs or row-shard state collective differ from the main fabric's",
                    );
                }
                cmb = rb;
            }
            if pf_rs_on {
                p.log.push(
                    format!(
                        "# rank {rank} DS41 prefill RS+AG: GR8_DS_PF_RS=1 (c < {}; Q {} rows, slice stride {} B; AG collectives {}..{}, {} B each)\n",
                        p.rs_k,
                        cm.quarter,
                        cm.slice_stride,
                        nc,
                        nc + p.rs_k - 1,
                        cm.slot_bytes3
                    )
                    .into_bytes(),
                );
            }
            // GR8_DS_PF_DIRECT=1: producers write the pinned send slots; RS slices go flat, so the QPs take 2 send SGEs
            if k.ds_pf_direct {
                if pf_rows <= mx {
                    return refuse("GR8_DS_PF_DIRECT=1 needs GR8_DS_PREFILL_W > MX");
                }
                if k.ds_subdump {
                    return refuse(
                        "GR8_DS_PF_DIRECT=1: GR8_DS_SUBDUMP reads Model::sc.part after the publish (not supported)",
                    );
                }
                p.pf_direct = true;
                p.log.push(
                    format!(
                        "# rank {rank} DS41 prefill direct publish: GR8_DS_PF_DIRECT=1 (send slots written by the producers; RS slices flat: {})\n",
                        if p.rs_k != 0 { "yes" } else { "no RS" }
                    )
                    .into_bytes(),
                );
            }
        }
        // the two --ips2 rails alone do not stripe: Fabric::init requires stripe_min > 0
        if k.ds_stripe {
            if i.n_ips2 != i.n_ips {
                return refuse("GR8_DS_STRIPE=1 needs --ips2 with one address per rank");
            }
            if cfg.ring || !cfg.vcap.is_empty() || cfg.bulk_bytes != 0 {
                return refuse("GR8_DS_STRIPE=1 needs a uniform, non-bulk mesh fabric");
            }
            cfg.stripe_min = k.ds_stripe_min as usize;
        }
        // no recv slot for a rank's own contribution: DS41 reads it from the send slot
        cfg.compact = true;
    }
    cfg.ncoll = nc;
    // --ring: the relay thread's own core
    cfg.relay_cpu = cpu(cfg.proxy_cpu, 8);
    // GR8_XTRACE=1: stage-0 trace on the decode fabric, or with GR8_XTRACE_PF=1 on the two prefill fabrics instead
    let xtrace_pf = k.xtrace_pf;
    cfg.xtrace = if xtrace_pf { 0 } else { 1 };
    let with_runs = |c: &mut Fc, r: &ps::Runs, pool: i32| {
        c.slot_bytes = r.slot_bytes;
        c.run1 = r.run1;
        c.run2 = r.run2;
        c.slot_bytes1 = r.slot_bytes1;
        c.slot_bytes2 = r.slot_bytes2;
        c.pool = pool;
    };
    {
        let mut mc = cfg.clone();
        if cm.run1 != 0 {
            with_runs(&mut mc, &cm, cm_pool);
        }
        if ds41 && p.rs_k != 0 {
            mc.run3 = cm.run3;
            mc.slot_bytes3 = cm.slot_bytes3;
            mc.ncoll = ps::ds41_main_ncoll(nc, p.rs_k, p.rowshard);
        }
        mc.sge2 = p.pf_direct;
        p.add(Kind::Main, &mc, false);
    }
    cfg.xtrace = 0;
    // GR8_DS_PF_LANES=2: lane B's own fabric (DS41 builds no prefill fabrics)
    if ds41 && p.pf_lanes_fab == 2 {
        if !i.s2mode || cmb.run1 == 0 {
            return refuse("GR8_DS_PF_LANES=2 needs --serve with W-row prefill (GR8_DS_PREFILL_W > MX)");
        }
        let mut mc = cfg.clone();
        with_runs(&mut mc, &cmb, cm_pool);
        if p.rs_k != 0 {
            mc.run3 = cmb.run3;
            mc.slot_bytes3 = cmb.slot_bytes3;
            mc.ncoll = ps::ds41_main_ncoll(nc, p.rs_k, p.rowshard);
        }
        mc.sge2 = p.pf_direct;
        mc.port = cfg.port + PORT_OFF[Kind::LaneB as usize];
        mc.proxy_cpu = cpu(cfg.proxy_cpu, 2);
        mc.xtrace = 0;
        p.add(Kind::LaneB, &mc, false);
    }
    let pf_rs = k.pf_rs;
    let n_ips = i.n_ips;
    // GR8_PF_STRIPE: prefill payload writes >= GR8_PF_STRIPE_MIN go out half on each rail (a three-rank ring: the same
    // port's device on the other PCIe root). Unset = on with --pf-ips2 or at a three-rank ring.
    let pf_stripe_cfg = |c: &mut Fc| -> Result<(), Str> {
        let ring3 = c.ring && n_ips == 3;
        let on = if k.pf_stripe < 0 { i.n_pf_ips2 != 0 || ring3 } else { k.pf_stripe != 0 };
        if i.n_pf_ips2 != 0 {
            c.pf_ips2 = true;
        }
        if !on {
            return Ok(());
        }
        let n_ips2 = if c.pf_ips2 { i.n_pf_ips2 } else { i.n_ips2 };
        if !ring3 && n_ips2 != n_ips {
            return refuse("GR8_PF_STRIPE=1 needs --pf-ips2 (or --ips2) with one address per rank");
        }
        c.stripe_min = k.pf_stripe_min as usize;
        Ok(())
    };
    let prefill = |p: &mut Plan, kind: Kind, cpu_off: i32| -> Result<(), Str> {
        let mut c = cfg.clone();
        c.slot_bytes = ps::pf_slot_bytes(mp, pf_rs, n_ips as i32, hid);
        c.ncoll = if pf_rs { 5 } else { 2 };
        if flash && i.pf_sp {
            c.slot_bytes = ps::pf_sp_slot_bytes(mp, n_ips as i32, hid);
            c.ncoll = 4;
        }
        c.port = cfg.port + PORT_OFF[kind as usize];
        pf_stripe_cfg(&mut c)?;
        c.proxy_cpu = cpu(cfg.proxy_cpu, cpu_off);
        c.xtrace = if xtrace_pf { 1 } else { 0 };
        p.add(kind, &c, false);
        Ok(())
    };
    if mp != 0 {
        prefill(p, Kind::Prefill, 1)?;
    }
    if i.lanes != 1 && i.lanes != 2 {
        return refuse("--pf-lanes must be 1 or 2");
    }
    let two = mp != 0 && i.lanes == 2;
    p.two = two;
    if two {
        prefill(p, Kind::Prefill2, 2)?;
    }
    if flash {
        // a one-collective fabric of a prefill lane, striped as the lanes are. Its proxy cut-off is one core higher
        // than the prefill fabrics' (proxy > off, not > off - 1), as the inline bring-up had it: at proxy == off it is
        // unpinned
        let lane_fabric = |p: &mut Plan, kind: Kind, cpu_off: i32, slot_bytes: usize| -> Result<(), Str> {
            let mut c = cfg.clone();
            c.ncoll = 1;
            c.slot_bytes = slot_bytes;
            c.port = cfg.port + PORT_OFF[kind as usize];
            c.proxy_cpu = if cfg.proxy_cpu > cpu_off { cfg.proxy_cpu - cpu_off } else { -1 };
            pf_stripe_cfg(&mut c)?;
            p.add(kind, &c, false);
            Ok(())
        };
        // the pool-ID exchange: default on for TP2 Flash prefill (unset = auto, 0 = off, 1 = on or refuse)
        let pool_mode = i.pool_owner_mode;
        let pool_fits = mp != 0 && nr == 2 && n_ips == 2 && !cfg.ring;
        let pool_conflict = if k.fpf_attn_p1 {
            Some("GR8_FPF_ATTN_P1")
        } else if k.fpf_dsa_p1 {
            Some("GR8_FPF_DSA_P1")
        } else if k.fpf_p1 {
            Some("GR8_FPF_P1")
        } else {
            None
        };
        if pool_mode == 1 && !pool_fits {
            return refuse("GR8_GLM53F_PF_POOL_OWNER=1 needs TP2 Flash prefill (not ring)");
        }
        if let (1, Some(c)) = (pool_mode, pool_conflict) {
            return Err(format!("GR8_GLM53F_PF_POOL_OWNER=1 conflicts with {c}").into_bytes());
        }
        p.pool_owner = pool_mode == 1 || (pool_mode == 2 && pool_fits && pool_conflict.is_none());
        if p.pool_owner {
            lane_fabric(p, Kind::PoolIds, 9, i.pool_payload)?;
            if two {
                lane_fabric(p, Kind::PoolIds2, 10, i.pool_payload)?;
            }
        }
        // GR8_PF_ROUTER_SPLIT=1: one route-table fabric per prefill lane
        p.route_split = i.route_split_mode == 1;
        if p.route_split {
            if !(mp != 0 && nr >= 2 && nr <= b.route_max_nr && n_ips == sz(nr)) {
                return refuse("GR8_PF_ROUTER_SPLIT=1 needs Flash prefill (--prefill) on 2 to 8 ranks");
            }
            lane_fabric(p, Kind::RouteSplit, 11, i.route_payload)?;
            if two {
                lane_fabric(p, Kind::RouteSplit2, 12, i.route_payload)?;
            }
        }
    }
    if mtp != 0 {
        // the MTP draft region: 3 collectives, decode-sized slots; its own core (prefill lane 2 has N-2)
        let mut c = cfg.clone();
        c.ncoll = 3;
        c.port = cfg.port + PORT_OFF[Kind::Draft as usize];
        c.proxy_cpu = cpu(cfg.proxy_cpu, 6);
        p.add(Kind::Draft, &c, false);
    }
    if ds41 && i.dsk_on {
        // the DSpark draft block: decode-sized slots (5 rows a session), the MTP draft's port (--mtp is refused with it)
        let mut c = cfg.clone();
        c.ncoll = b.ds_draft_ncoll;
        c.slot_bytes = p.dec_slot_bytes;
        if c.ncoll > 0 && i.dsp_coll_max > c.slot_bytes {
            return refuse("DSpark draft fabric: a collective is larger than the decode slot");
        }
        c.port = cfg.port + PORT_OFF[Kind::DSpark as usize];
        c.proxy_cpu = cpu(cfg.proxy_cpu, 6);
        c.xtrace = if k.ds_xtrace { 1 } else { 0 };
        p.add(Kind::DSpark, &c, false);
    }
    if i.dfl {
        // --dflash: the drafter's region, decode-sized slots; its own core (not N-2 or N-6)
        let mut c = cfg.clone();
        c.ncoll = i.dfl_ncoll;
        c.port = cfg.port + PORT_OFF[Kind::Dflash as usize];
        c.proxy_cpu = cpu(cfg.proxy_cpu, 7);
        p.add(Kind::Dflash, &c, false);
    }
    if nr == 4 && i.qp_roll_s != 0 {
        // GR8_PF_QP_ROLL: a ring of [lo | hi] pair half images, one-peer jobs
        let mut c = cfg.clone();
        c.vcap.clear();
        c.vdst_k.clear();
        c.bulk_bytes = ps::qw_bulk_bytes(i.qp_wire_e, b.qw_half);
        c.bulk_flags = ps::qw_bulk_flags(i.qp_wire_e);
        c.port = cfg.port + PORT_OFF[Kind::QpWire as usize];
        c.proxy_cpu = cpu(cfg.proxy_cpu, 5);
        p.add(Kind::QpWire, &c, false);
    }
    // CP (GR8_CP, GR8_CP_R, GR8_CP_S3, GR8_CP_MMAX): its own fabric, cplayout.h's variable layout
    if i.cp_build {
        p.cp_on = k.cp != 0;
        p.cp_r = k.cp_r as i32;
        p.cp_s3 = i32::from(k.cp_s3);
        let mut cp_mmax = i.drafts + 1;
        for &m in slice(i.row_counts, i.n_row_counts) {
            cp_mmax = cp_mmax.max(m);
        }
        // S2: any step may run the CP block; GR8_CP_STEP_ROWS=n caps the steps past R at n rows
        if i.s2mode {
            let full = mx.min((i.s2sess * (mtp + 1)).max(i.chunk));
            match cp_rows::cp_step_rows(k.cp_step_rows, cp_mmax, full) {
                Some(n) => p.cp_step = n,
                None => {
                    p.cp_step = -1;
                    return Err(format!(
                        "GR8_CP_STEP_ROWS={}: at least {cp_mmax} (the rows of a solo or prompt step past R: the verify width and --chunk)",
                        k.cp_step_rows
                    )
                    .into_bytes());
                }
            }
            cp_mmax = cp_mmax.max(if p.cp_step != 0 { p.cp_step } else { full });
        } else if k.cp_step_rows != 0 {
            return refuse("GR8_CP_STEP_ROWS needs --serve");
        }
        if k.has_cp_mmax {
            cp_mmax = cp_mmax.max(k.cp_mmax as i32);
        }
        p.cp_mmax = cp_mmax;
        if p.cp_r < 0 || p.cp_r % 64 != 0 {
            return refuse("GR8_CP_R must be a multiple of 64");
        }
        // GR8_CP_R2: a session may take a smaller R; every step past it runs the CP block
        p.cp_r2 = if k.has_cp_r2 { k.cp_r2 as i32 } else { 0 };
        if p.cp_r2 != 0
            && (p.cp_r2 < 64 || p.cp_r2 % 64 != 0 || p.cp_r2 >= p.cp_r || !p.cp_on || !i.s2mode || i.rt_win == 0)
        {
            return refuse(
                "GR8_CP_R2 must be a multiple of 64 below GR8_CP_R, with GR8_CP on, --serve and GR8_RT_LOCAL",
            );
        }
        p.cp_step_r = if p.cp_r2 != 0 { p.cp_r2 } else { p.cp_r };
        let d = &b.cp;
        p.cp = ps::cp_colls(
            slice(i.full_index, i.n_full_index),
            n_ips as i32,
            cp_mmax,
            d.heads,
            d.kv_rank,
            d.qk_dim,
            d.topk,
        );
        if p.cp_on {
            let mut c = cfg.clone();
            c.vcap = p.cp.cap.clone();
            c.vdst_k = p.cp.dst_k.clone();
            c.port = cfg.port + PORT_OFF[Kind::Cp as usize];
            c.proxy_cpu = cpu(cfg.proxy_cpu, 3);
            p.add(Kind::Cp, &c, false);
        }
    }
    // the fabrics the built model completes. The gathers (GR8_QA_SPLIT_CPU moves their proxy; main and draft steps
    // never run at once, so both take N-9)
    let gather = |p: &mut Plan, kind: Kind| {
        let mut c = cfg.clone();
        c.vcap.clear();
        c.vdst_k.clear();
        c.port = cfg.port + PORT_OFF[kind as usize];
        c.proxy_cpu = if k.has_qa_split_cpu { k.qa_split_cpu as i32 } else { cpu(cfg.proxy_cpu, 9) };
        p.add(kind, &c, true);
    };
    if i.gather {
        gather(p, Kind::Gather);
    }
    if p.cp_on && mp != 0 {
        // CP prefill staging pools: a bulk-mode fabric
        let mut c = cfg.clone();
        c.vcap.clear();
        c.vdst_k.clear();
        c.bulk_flags = 4;
        c.port = cfg.port + PORT_OFF[Kind::CpStage as usize];
        c.proxy_cpu = cpu(cfg.proxy_cpu, 4);
        p.add(Kind::CpStage, &c, true);
    }
    if i.draft_gather {
        gather(p, Kind::DraftGather);
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------------ the C ABI

/// g6plan_result: the view and the storage it points into
pub struct PlanResult {
    _refusal: CString,
    _log: Vec<CString>,
    _log_p: Vec<*const c_char>,
    _vcaps: Vec<Vec<u32>>,
    _vdsts: Vec<Vec<u8>>,
    _cp: CpColls,
    view: CPlan,
}

fn entry(f: &Planned, vcaps: &mut Vec<Vec<u32>>, vdsts: &mut Vec<Vec<u8>>) -> Entry {
    let c = &f.cfg;
    vcaps.push(c.vcap.clone());
    vdsts.push(c.vdst_k.clone());
    let (vcap, vdst_k) = (vcaps[vcaps.len() - 1].as_ptr(), vdsts[vdsts.len() - 1].as_ptr());
    let (n_vcap, n_vdst_k) = (c.vcap.len(), c.vdst_k.len());
    Entry {
        kind: f.kind as u32,
        late: f.late,
        use_pf_ips2: c.pf_ips2,
        port_off: PORT_OFF[f.kind as usize],
        cfg: Cfg {
            rank: c.rank,
            stripe_min: c.stripe_min,
            slot_bytes: c.slot_bytes,
            run1: c.run1,
            run2: c.run2,
            slot_bytes1: c.slot_bytes1,
            slot_bytes2: c.slot_bytes2,
            run3: c.run3,
            slot_bytes3: c.slot_bytes3,
            pool: c.pool,
            sge2: c.sge2,
            compact: c.compact,
            ncoll: c.ncoll,
            port: c.port,
            proxy_cpu: c.proxy_cpu,
            vcap,
            n_vcap,
            vdst_k,
            n_vdst_k,
            bulk_bytes: c.bulk_bytes,
            bulk_flags: c.bulk_flags,
            xtrace: c.xtrace,
            ring: c.ring,
            relay_cpu: c.relay_cpu,
        },
    }
}

/// Plans the fabrics. Free the result with g6plan_free.
///
/// # Safety
/// `i` is valid, with every pointer valid for its count and every text null or a C string.
#[no_mangle]
pub unsafe extern "C" fn g6plan_run(i: *const In) -> *mut PlanResult {
    let mut p = Plan { i_late: -1, pf_lanes_fab: 1, ..Default::default() };
    let r = plan(&*i, &mut p);
    let refusal = match r {
        Ok(()) => Str::new(),
        Err(e) => {
            p.fabrics.clear();
            e
        }
    };
    let refusal = cs(&refusal);
    let log: Vec<CString> = p.log.iter().map(|l| cs(l)).collect();
    let log_p: Vec<*const c_char> = log.iter().map(|l| l.as_ptr()).collect();
    let (mut vcaps, mut vdsts) = (Vec::new(), Vec::new());
    // SAFETY: zero is a valid Entry (integers, false, null pointers)
    let mut entries: [Entry; KINDS] = std::mem::zeroed();
    debug_assert!(p.fabrics.len() <= KINDS, "more fabrics than entries");
    for (e, f) in entries.iter_mut().zip(&p.fabrics) {
        *e = entry(f, &mut vcaps, &mut vdsts);
    }
    let cp = std::mem::take(&mut p.cp);
    let view = CPlan {
        refusal: refusal.as_ptr(),
        log: log_p.as_ptr(),
        n_log: log_p.len(),
        n: p.fabrics.len() as i32,
        entries,
        dec_slot_bytes: p.dec_slot_bytes,
        pf_rows: p.pf_rows,
        tail: p.tail,
        i_late: p.i_late,
        l20_tail: p.l20_tail,
        rs_k: p.rs_k,
        quarter: p.quarter,
        slice_stride: p.slice_stride,
        rowshard: p.rowshard,
        y_mxfp8: p.y_mxfp8,
        pf_direct: p.pf_direct,
        pf_lanes_fab: p.pf_lanes_fab,
        two: p.two,
        pool_owner: p.pool_owner,
        route_split: p.route_split,
        cp_on: p.cp_on,
        cp_r: p.cp_r,
        cp_r2: p.cp_r2,
        cp_s3: p.cp_s3,
        cp_mmax: p.cp_mmax,
        cp_step: p.cp_step,
        cp_step_r: p.cp_step_r,
        cp: CCp {
            nr: cp.nr,
            nl: cp.nl,
            mmax: cp.mmax,
            q_row: cp.q_row,
            cand_row: cp.cand_row,
            part_row: cp.part_row,
            q: cp.q.as_ptr(),
            n_q: cp.q.len(),
            cand: cp.cand.as_ptr(),
            n_cand: cp.cand.len(),
            part: cp.part.as_ptr(),
            n_part: cp.part.len(),
            cap: cp.cap.as_ptr(),
            dst_k: cp.dst_k.as_ptr(),
            ncoll: cp.cap.len(),
        },
    };
    Box::into_raw(Box::new(PlanResult {
        _refusal: refusal,
        _log: log,
        _log_p: log_p,
        _vcaps: vcaps,
        _vdsts: vdsts,
        _cp: cp,
        view,
    }))
}

/// The plan; it lives until g6plan_free.
///
/// # Safety
/// `p` comes from g6plan_run.
#[no_mangle]
pub unsafe extern "C" fn g6plan_view(p: *const PlanResult) -> *const CPlan {
    &(*p).view
}

/// # Safety
/// `p` comes from g6plan_run, once.
#[no_mangle]
pub unsafe extern "C" fn g6plan_free(p: *mut PlanResult) {
    if !p.is_null() {
        drop(Box::from_raw(p));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module_info::MODEL_GLM53;

    /// the fabrics each model's g6m4 can create; Draft and DSpark never both (the parse refuses --mtp with DSpark)
    fn kinds_of(model: u32) -> Vec<Kind> {
        use Kind::*;
        let mut v = vec![Main, Prefill, Prefill2, Draft, Dflash, QpWire, Cp, CpStage];
        match model {
            MODEL_GLM53 => v.extend([Gather, DraftGather]),
            MODEL_GLM53F => v.extend([PoolIds, PoolIds2, RouteSplit, RouteSplit2]),
            MODEL_DS41 => v.extend([LaneB, DSpark]),
            _ => unreachable!(),
        }
        v
    }

    #[test]
    fn every_kind_belongs_to_a_model() {
        let mut seen = [false; KINDS];
        for model in [MODEL_GLM53, MODEL_GLM53F, MODEL_DS41] {
            for k in kinds_of(model) {
                seen[k as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "a Kind missing from kinds_of: {seen:?}");
    }

    #[test]
    fn ports_distinct_per_model() {
        for model in [MODEL_GLM53, MODEL_GLM53F, MODEL_DS41] {
            let ks = kinds_of(model);
            for (x, &a) in ks.iter().enumerate() {
                for &b in &ks[x + 1..] {
                    if (a, b) == (Kind::Draft, Kind::DSpark) || (a, b) == (Kind::DSpark, Kind::Draft) {
                        continue;
                    }
                    let (pa, pb) = (PORT_OFF[a as usize], PORT_OFF[b as usize]);
                    assert!((pa - pb).abs() >= 4, "model {model}: {a:?} +{pa} and {b:?} +{pb} share bootstrap ports");
                }
            }
        }
    }

    #[test]
    fn proxy_cores() {
        assert_eq!((cpu(13, 9), cpu(9, 9), cpu(8, 9)), (4, 0, -1));
    }
}
