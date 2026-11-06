// module_info.h: the build's description matches its constants, and each model has its capabilities. Built once per
// model flag (none, -DG6_MODEL_FLASH=1, -DG6_MODEL_DS41=1) by scripts/host_tests.sh. Host only.
#include "../../module_info.h"
#include "../s2/prompt_rule.h"
#include "../s2/scheduler.h"
#include "../s2/sessions.h"
#include "../s2_batch.h"
#include <cstdio>
#include <cstring>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

// the description is a constant expression: the slot counts below are array sizes in the scheduler and the planner
static_assert(g6mod::INFO.abi == G6_MODULE_ABI, "abi");
static_assert(g6s2::MAX_SESSIONS == unsigned(g6mod::INFO.max_sessions) && s2::S2N_SLOTS == g6mod::INFO.max_sessions,
              "slot counts");

int main()
{
    const G6ModuleInfo& m = g6mod::INFO;
    CHECK(m.nr == G6_NR && m.mx == G6_MX && m.s2n == g6rt::S2N);
    CHECK(m.hidden == geom::HID && m.layers == geom::L_MAIN && m.vocab == geom::VOCAB && m.experts == geom::NEXP &&
          m.topk == geom::TOPK);
    CHECK(m.kv_row_bytes == g6kv::ROW && std::strcmp(m.kv_format, g6kv::NAME) == 0 && m.kv_page == 64);
    CHECK(m.abi == 2 && m.dfl_block == 8 && m.pack_members == 8);
#if G6_MODEL_DS41
    CHECK(m.model == G6_MODEL_ID_DS41 && std::strcmp(m.name, "ds41") == 0 && m.max_sessions == 32 && m.mtp_layers == 3);
    CHECK(m.caps == (G6_CAP_MTP0 | G6_CAP_PACK | G6_CAP_PARK | G6_CAP_PROMPT_DECODE | G6_CAP_DSPARK));
#elif G6_MODEL_FLASH
    CHECK(m.model == G6_MODEL_ID_GLM53F && std::strcmp(m.name, "glm-5.3-flash") == 0 && m.max_sessions == 8);
    CHECK(m.caps == (G6_CAP_DFLASH2 | G6_CAP_FLASH_COST | G6_CAP_KDA | G6_CAP_GRAMMAR));
#else
    CHECK(m.model == G6_MODEL_ID_GLM53 && std::strcmp(m.name, "glm-5.3") == 0 && m.max_sessions == 8 && m.mtp_layers == 1);
    CHECK(m.caps == (G6_CAP_DFLASH2 | G6_CAP_GRAMMAR));
#endif
    // the scheduler's model rules read the capabilities: each refusal by its own message, nothing else refused
    auto refusal = [](const g6s2::Config& c) -> std::string {
        try { c.validate(); } catch (const std::exception& e) { return e.what(); }
        return "";
    };
    g6s2::Config ok;
    ok.epoch = 1; ok.pool_rows = 64 * 64; ok.mtp = 3; ok.max_rows = 32; ok.max_sessions = 1; ok.prefill_rows = 64;
    CHECK(refusal(ok).empty());
    g6s2::Config c = ok;
    c.mtp = 0;
    CHECK(refusal(c) == ((m.caps & G6_CAP_MTP0) ? "" : "bad session/row limit"));
    c = ok;
    c.c8_pack = true;
    CHECK(refusal(c) == ((m.caps & G6_CAP_PACK) ? "" : "c8 prefill pack is DS41 only"));
    c = ok;
    c.park = true;
    CHECK(refusal(c) == ((m.caps & G6_CAP_PARK) ? "" : "parked slots are DS41 only"));
    // the session prompt rule: with prefill_min 0 every prompt decodes where prompts decode by default, elsewhere only
    // prompt logprobs keep a prompt on the decode path
    const g6s2::Sessions none(ok, "", false, 4), pf(ok, "", false, 4, 64);
    g6rt::ServeReq r;
    r.prompt.assign(5000, 7); r.n_gen = 8;
    const bool dflt = (m.caps & G6_CAP_PROMPT_DECODE) != 0;
    CHECK(none.payload(r).decode_prompt == dflt && !pf.payload(r).decode_prompt);
    r.plp = 0;
    CHECK(none.payload(r).decode_prompt && pf.payload(r).decode_prompt);
    std::printf("module_info_test (%s, TP%d, %d rows, %d slots, KV %s): %s\n", m.name, m.nr, m.mx, m.max_sessions, m.kv_format,
                fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
