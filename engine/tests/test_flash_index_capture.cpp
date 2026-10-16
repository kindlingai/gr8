#include "../runtime/flash_index_capture.h"
#include <cassert>
#include <functional>
#include <iostream>
using namespace flash_index_capture;
static std::vector<int> ids_for(int pos) {
    std::vector<int> ids(STRIDE, -1);
    const int count = pos + 1, visible = count / KP, keep = std::min(visible, POOLS);
    int at = 0;
    // Select the last visible pools, not just the first 512.
    for (int p = visible - keep; p < visible; ++p)
        for (int k = 0; k < KP; ++k) ids[at++] = p * KP + k;
    for (int t = count - count % KP; t < count; ++t) ids[at++] = t;
    return ids;
}
int main(int argc, char** argv) {
    int cases = 0;
    for (int context : {1, 2, 3, 4, 5, 2047, 2048, 2051, 2052, 32771, 131072, 300000}) {
        const int pos = context - 1;
        auto ids = ids_for(pos);
        const auto got = pack(context, 1, &pos, ids.data());
        assert(got[0] == pos && got.size() == ROW_WORDS);
        const int keep = std::min(context / KP, POOLS);
        for (int j = 0; j < POOLS; ++j)
            assert(got[1 + j] == (j < keep ? context / KP - keep + j : -1));
        ++cases;
    }
    auto rejects = [&](std::function<void()> f) {
        bool rejected = false;
        try { f(); } catch (const std::runtime_error&) { rejected = true; }
        assert(rejected); ++cases;
    };
    const int context = 300000, pos = context - 1;
    for (int type = 0; type < 6; ++type) {
        auto ids = ids_for(pos);
        if (type == 0) ids[0] = -1;
        if (type == 1) ids[0] += 1;
        if (type == 2) ids[1] += 1;
        if (type == 3) for (int k = 0; k < KP; ++k) ids[KP + k] = ids[k];
        if (type == 4) for (int k = 0; k < KP; ++k) ids[(POOLS-1)*KP+k] = context+k;
        if (type == 5) ids[STRIDE-1] = 0;
        rejects([&] { pack(context, 1, &pos, ids.data()); });
    }
    auto ids = ids_for(2050);const int tail_pos = 2050;ids[2048]++;
    rejects([&] { pack(2051, 1, &tail_pos, ids.data()); });
    rejects([&] { pack(context, 0, &pos, ids.data()); });
    rejects([&] { pack(context, 1, nullptr, ids.data()); });
    const int wrong_pos = pos-1;
    rejects([&] { pack(context, 1, &wrong_pos, ids.data()); });
    if (argc == 2) {
        // Small multi-layer, multi-block fixture consumed by the independent Python checker.
        Writer writer;
        for (int end : {4, 9}) for (int layer : {3, 7}) {
            const int begin = end == 4 ? 0 : 4, rows = end - begin;
            std::vector<int> ps(rows), is;
            for (int r = 0; r < rows; ++r) {
                ps[r] = begin+r;auto v = ids_for(ps[r]);is.insert(is.end(),v.begin(),v.end());
            }
            writer.write(argv[1],layer,end,rows,ps.data(),is.data());
        }
    }
    std::cout << "PASS flash pool capture host checks=" << cases << "\n";
}
