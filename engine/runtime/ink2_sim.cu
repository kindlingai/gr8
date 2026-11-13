// m4sim_ink2: Inkling-2-Small with every rank of a TP width on one GPU (ink2_step.cuh). The ranks' partial sums meet
// in one buffer in rank order, as they do after the fabric exchange of g6m4_ink2 (ink2_m4.cu), so both give the same
// tokens. The checkpoint (a directory with config.json and its safetensors files) is read in place, one tensor at a
// time; a miniature checkpoint and a subset of the layers run through the same code.
//
//   m4sim_ink2 --tp 4 --v2-dense DIR [--layers all|0,5,..] [--tokens 1,2,3] [--gen 16] [--sessions 1] [--chunk 8]
//              [--temp 0 --top-p 1 --seed 1234] [--top-p-cands 256] [--cap 4096]
//
// --tp is taken by widths.cpp, which picks this width's instance. Every session runs the same prompt: the prompt in
// steps of --chunk rows, then --gen steps of one row a session. Prints each session's generated tokens.
#include "ink2_step.cuh"

#ifndef G6_NR
#define G6_NR 6
#endif

int main(int argc, char** argv)
{
    using namespace ink2rt;
    std::string dir, layers = "all", tokens;
    int gen = 16, sessions = 1, chunk = 8, nc = 256, cap = 4096;
    float temp = 0.f, top_p = 1.f;
    uint32_t seed = 1234;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        const auto val = [&]() -> std::string {
            if (i + 1 >= argc) die(a + " needs a value");
            return argv[++i];
        };
        if (a == "--v2-dense" || a == "--model" || a == "--packs") dir = val();
        else if (a == "--layers") layers = val();
        else if (a == "--tokens") tokens = val();
        else if (a == "--gen") gen = atoi(val().c_str());
        else if (a == "--sessions" || a == "--s2-sessions") sessions = atoi(val().c_str());
        else if (a == "--chunk") chunk = atoi(val().c_str());
        else if (a == "--top-p-cands") nc = atoi(val().c_str());
        else if (a == "--cap") cap = atoi(val().c_str());
        else if (a == "--temp") temp = static_cast<float>(atof(val().c_str()));
        else if (a == "--top-p") top_p = static_cast<float>(atof(val().c_str()));
        else if (a == "--seed") seed = static_cast<uint32_t>(strtoul(val().c_str(), nullptr, 10));
        else if (a == "--graph" || a == "--no-graph") {}   // the step is eager
        else die("unknown option " + a);
    }
    if (nc < 0 || nc > 2048 / G6_NR) die("--top-p-cands must be 0.." + std::to_string(2048 / G6_NR));
    if (dir.empty()) die("name the checkpoint: --v2-dense DIR");
    if (sessions < 1 || sessions > 8 || chunk < 1 || chunk > 32 || gen < 0 || cap < 64) die("--sessions 1..8, --chunk 1..32, --gen >= 0, --cap >= 64");
    knob::init();

    Options o;
    o.nr = G6_NR;
    o.rank0 = 0;
    o.nloc = G6_NR;
    o.slots = sessions;
    o.max_rows = std::max(chunk, sessions);
    o.cap = cap;
    o.nc = std::max(nc, 1);
    o.layers = layers;
    Engine e;
    e.init(dir, o);

    std::vector<int> prompt;
    {
        std::stringstream ss(tokens);
        std::string item;
        while (std::getline(ss, item, ',')) prompt.push_back(atoi(item.c_str()));
        if (prompt.empty()) prompt.push_back(geom::TOK_BOS < e.c.vocab ? static_cast<int>(geom::TOK_BOS) : 1);
        for (int t : prompt)
            if (t < 0 || t >= e.c.vocab) die("--tokens: " + std::to_string(t) + " is not a token of the vocabulary");
    }
    const int P = static_cast<int>(prompt.size());
    if (P + gen > cap) die("the prompt and --gen pass --cap");

    std::vector<RowOut> out;
    std::vector<int> next(static_cast<size_t>(sessions));
    std::vector<std::vector<int>> said(static_cast<size_t>(sessions));
    const float inv_temp = temp > 0.f ? 1.f / temp : 0.f;
    for (int s = 0; s < sessions; ++s)
    {
        e.reset_slot(s);
        for (int p = 0; p < P; p += chunk)
        {
            const int n = std::min(chunk, P - p);
            ink2::RowsHost rh;
            rh.add(s, p, n);
            std::vector<RowSample> sm(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) sm[i] = RowSample{inv_temp, top_p, seed, static_cast<uint32_t>(p + i + 1)};
            e.forward(rh, &prompt[static_cast<size_t>(p)], sm.data(), out);
            e.commit();
            next[s] = out[static_cast<size_t>(n) - 1].tok;
        }
    }
    for (int g = 0; g < gen; ++g)
    {
        ink2::RowsHost rh;
        std::vector<RowSample> sm(static_cast<size_t>(sessions));
        for (int s = 0; s < sessions; ++s)
        {
            said[s].push_back(next[s]);
            rh.add(s, P + g, 1);
            sm[s] = RowSample{inv_temp, top_p, seed, static_cast<uint32_t>(P + g + 1)};
        }
        if (g + 1 == gen) break;
        e.forward(rh, next.data(), sm.data(), out);
        e.commit();
        for (int s = 0; s < sessions; ++s) next[s] = out[s].tok;
    }
    for (int s = 0; s < sessions; ++s)
    {
        printf("session %d tokens:", s);
        for (int t : said[s]) printf(" %d", t);
        printf("\n");
    }
    return 0;
}
