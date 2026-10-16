#include "../runtime/draft_head_dump.h"
#include <vector>
#include <string>
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    draft_head_dump::Writer off;
    // OFF: no pointer dereference, allocation, file, or footer.
    off.call(0, 0, 0, 999, true, 0, nullptr, nullptr);
    off.finish();
    std::vector<float> h(2 * 4096, 0.125f);
    draft_head_dump::Row r[] = {{10, 20, 21, 12, 42, 0, 0, 0, 0, -1},
                                {11, 21, 22, 13, 42, 0, 0, 0, 0, -2}};
    for (int rank = 0; rank < 2; ++rank) {
        draft_head_dump::Writer w;
        w.open(argv[1], 2, rank, 4096, 154880, 8, 3, false);
        w.call(0, 1, 0, 2, false, 12, r, h.data());
        draft_head_dump::Row chain = {12, 22, 23, 14, 42, 0, 0, 0, 0, -3};
        w.call(0, 2, 1, 1, true, 13, &chain, h.data());
        chain.pos++; chain.input++; chain.output++; chain.predict_pos++;
        w.call(0, 3, 2, 1, true, 14, &chain, h.data());
        w.finish();
    }
    puts("PASS writer: OFF null pointers, two ranks, first form + two chain calls");
}
