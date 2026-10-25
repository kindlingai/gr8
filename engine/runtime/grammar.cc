// The XGrammar side of grammar.h. Built by the host compiler when G6_XGRAMMAR names an XGrammar build (build_m4.sh).
#include "grammar.h"

#include <xgrammar/xgrammar.h>
#include <picojson.h>

#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace g6rt
{
namespace
{
constexpr int64_t CACHE_BYTES = 256ll << 20;
struct Boot
{
    std::optional<xgrammar::TokenizerInfo> info;
    std::optional<xgrammar::GrammarCompiler> compiler;
    int vocab = 0, words = 0;
};
Boot& boot()
{
    static Boot b;
    return b;
}
std::string slurp(const std::string& path)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error(path + ": cannot read");
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
picojson::value parse(const std::string& text, const std::string& what)
{
    picojson::value v;
    const std::string err = picojson::parse(v, text);
    if (!err.empty()) throw std::runtime_error(what + ": " + err);
    return v;
}
}   // namespace

struct GrammarState
{
    xgrammar::GrammarMatcher m;
    int arm;
    bool armed, broken = false;
    GrammarState(const xgrammar::CompiledGrammar& g, int a) : m(g), arm(a), armed(a < 0) {}
};
void GrammarDeleter::operator()(GrammarState* g) const { delete g; }

std::string grammar_init(const std::string& dir, int vocab)
{
    try
    {
        const std::string backend = slurp(dir + "/tokenizer.json");
        const picojson::value tj = parse(backend, dir + "/tokenizer.json");
        // the vocabulary in id order: the model's tokens, then the added (special) tokens; ids past them stay empty
        std::vector<std::string> v(vocab);
        for (const auto& kv : tj.get("model").get("vocab").get<picojson::object>())
        {
            const int id = (int)kv.second.get<double>();
            if (id >= 0 && id < vocab) v[id] = kv.first;
        }
        for (const auto& t : tj.get("added_tokens").get<picojson::array>())
        {
            const int id = (int)t.get("id").get<double>();
            if (id >= 0 && id < vocab) v[id] = t.get("content").get<std::string>();
        }
        std::vector<int32_t> stop;
        const picojson::value gc = parse(slurp(dir + "/generation_config.json"), dir + "/generation_config.json");
        const picojson::value& eos = gc.get("eos_token_id");
        if (eos.is<double>()) stop.push_back((int32_t)eos.get<double>());
        else if (eos.is<picojson::array>())
            for (const auto& e : eos.get<picojson::array>()) stop.push_back((int32_t)e.get<double>());
        if (stop.empty()) throw std::runtime_error(dir + "/generation_config.json: no eos_token_id");
        const picojson::value meta = parse(xgrammar::TokenizerInfo::DetectMetadataFromHF(backend), "tokenizer metadata");
        auto& b = boot();
        b.info.emplace(v, (xgrammar::VocabType)(int)meta.get("vocab_type").get<double>(), vocab, stop,
                       meta.get("add_prefix_space").get<bool>());
        // the cache keeps compiled schemas for later requests. A request picks its schema, and one short `pattern`
        // compiles to 41 MB, so the cache is capped: past the cap, the least recently used schema is compiled again
        b.compiler.emplace(*b.info, 8, true, CACHE_BYTES);
        b.vocab = vocab;
        b.words = xgrammar::GetBitmaskSize(vocab);
        return "";
    }
    catch (const std::exception& e)
    {
        return e.what();
    }
}
bool grammar_ready() { return boot().compiler.has_value(); }
int grammar_words() { return boot().words; }

Grammar grammar_open(const GrammarSpec& spec)
{
    auto& b = boot();
    if (!b.compiler) throw std::runtime_error("the engine was started without --grammar-dir");
    // json_object is any JSON object. The layout is a 2-space pretty print, the one GLM-5.3 writes unasked: then the
    // drafters' drafts (which see no grammar) mostly fit, where a compact layout cost 13% of decode in rejected drafts,
    // and the model cannot pad a value with the tabs and newlines a free-whitespace grammar allows. Keys in any order
    // (the front end checks the required ones)
    const std::string schema = spec.kind == GrammarSpec::JSON ? std::string(R"({"type": "object"})") : spec.schema;
    const xgrammar::CompiledGrammar g =
        b.compiler->CompileJSONSchema(schema, false, 2, std::nullopt, spec.strict, std::nullopt, true);
    return Grammar(new GrammarState(g, spec.arm));
}

void grammar_commit(GrammarState& g, const int* ids, int n)
{
    for (int i = 0; i < n && !g.broken; ++i)
    {
        if (!g.armed)
        {
            g.armed = ids[i] == g.arm;
            continue;
        }
        if (g.m.IsTerminated() || !g.m.AcceptToken(ids[i])) g.broken = true;   // a token the mask allowed: never, but
    }                                                                          // the reply then goes on unconstrained
}

void grammar_masks(GrammarState& g, const int* drafts, int nd, int* on, uint32_t* out)
{
    const int words = boot().words;
    int64_t shape[1] = {words};
    bool armed = g.armed;
    int accepted = 0;
    for (int i = 0; i <= nd; ++i)
    {
        on[i] = 0;
        if (g.broken || g.m.IsTerminated()) continue;
        if (armed)
        {
            DLTensor t{out + (size_t)i * words, DLDevice{kDLCPU, 0}, 1, DLDataType{kDLInt, 32, 1}, shape, nullptr, 0};
            g.m.FillNextTokenBitmask(&t);
            on[i] = 1;
        }
        if (i == nd) break;
        if (!armed) armed = drafts[i] == g.arm;
        else if (g.m.AcceptToken(drafts[i])) ++accepted;
        else
        {   // the row's own mask refuses this draft, so the verify rejects it and the rows after it are dropped
            for (int k = i + 1; k <= nd; ++k) on[k] = 0;
            break;
        }
    }
    if (accepted) g.m.Rollback(accepted);
}
}   // namespace g6rt
