//! Drama 3 (`fish_omni`) TP=1 trellis expert image planner.
//!
//! Plans the expert half of `qw38.rs::plan` at `nr = 1, rank = 0`:
//!   - `H = 2048`, `L_MAIN = 40`, `NH = 4` (512 / 128), `NEXP = 256` routed experts + shared expert at row 256
//!   - Emits `Lnnn.tr3.{fc1,fc2,sg,v1b,v2,v3b,table,kmix}` via `v2plan::v2_layer` for every layer present in
//!     the width manifest (partial manifests are supported so harvested layers load natively while unharvested
//!     layers fall back to reference experts in `tts_weights.h`).
//!   - No dense passthrough: dense/vocab/tower tensors come from `tts_plan.rs`.

use crate::layout::{Group, Layout, Model};
use crate::plan::Plan;
use crate::v2plan::{self, Builder, Slices, V2Image};
use std::path::Path;

const ALIGN: u64 = 4096;
pub const H: usize = 2048;
pub const L_MAIN: usize = 40;
pub const NH: usize = 4;
pub const NEXP: usize = 256;

fn resolve_manifest_path(release_dir: &str, manifest_name: &str) -> Result<String, String> {
    if Path::new(manifest_name).is_file() {
        return Ok(manifest_name.to_string());
    }
    let fname = if manifest_name.ends_with(".json") {
        manifest_name.to_string()
    } else {
        format!("exlr8-manifest-{manifest_name}.json")
    };
    for cand in [
        format!("{release_dir}/manifests/{fname}"),
        format!("{release_dir}/{fname}"),
        format!("{release_dir}/manifests/{manifest_name}"),
        format!("{release_dir}/{manifest_name}"),
    ] {
        if Path::new(&cand).is_file() {
            return Ok(cand);
        }
    }
    Err(format!("dr3: cannot find manifest {manifest_name:?} under {release_dir}"))
}

/// Plans the TP=1 trellis expert image for Drama 3 from an exlr8 release directory.
pub fn plan(release_dir: &str, manifest_name: &str, layout_path: Option<&str>) -> Result<V2Image, String> {
    let nr = 1usize;
    let rank = 0usize;
    let man_path = resolve_manifest_path(release_dir, manifest_name)?;
    let man = v2plan::width_manifest(&man_path)?;
    let layout = match layout_path.filter(|s| !s.is_empty()) {
        Some(p) => {
            let l = Layout::read(p)?;
            let m = Model::release(&man)?;
            l.check(&m)?;
            if l.nr() != nr {
                return Err(format!("dr3: layout {p} has {} members, want TP 1", l.nr()));
            }
            Some(l)
        }
        None => None,
    };

    let slices = Slices::open(release_dir, nr, &[rank])?;
    let wfile = |l: usize, k: u8| v2plan::width_file(release_dir, l, k);

    let mut b = Builder::new();
    let mut planned = 0usize;
    for l in 0..L_MAIN {
        let Some(w) = v2plan::widths_nh(&man, l, NH)? else {
            continue;
        };
        if w.down.len() != NEXP {
            return Err(format!("dr3: layer {l} has {} routed experts, want {NEXP}", w.down.len()));
        }
        let own = |e: usize| -> Result<Vec<usize>, String> {
            match &layout {
                Some(ly) => ly.held(Group::Halves(l, e), rank),
                None => Ok(v2plan::formula_halves_nh(e, l, nr, rank, NH)),
            }
        };
        v2plan::v2_layer(&mut b, l, &w, &slices, &wfile, &own)?;
        planned += 1;
    }

    if planned == 0 {
        return Err(format!("dr3: manifest {man_path} listed 0 Drama 3 expert layers in 0..{L_MAIN}"));
    }

    let bytes = b.cur.div_ceil(ALIGN) * ALIGN;
    let plan = Plan::from_segs(b.files, b.segs, b.mems, b.gens, bytes)?;
    Ok(V2Image { rank, tensors: b.tensors, bytes, plan, cache: None })
}
