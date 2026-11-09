#include "../runtime/ds41_v2_attn_pipe.h"
#include <cassert>
#include <string>
int main() {
    ds41::v2::AttnPipe p;
    assert(!p.on);
    FILE* f = tmpfile(); assert(f);
    p.select(42, 0, f); p.select(42, 0, f);
    p.select(43, 1, f); p.select(42, 0, f);
    p.on = true; p.select(44, 0, f); p.select(44, 0, f);
    assert(p.seen.size() == 2);
    rewind(f); std::string s; char b[256];
    while (fgets(b, sizeof b, f)) s += b;
    fclose(f);
    assert(s == "# ds41_v2_attn_pipe request=42 on=0\n# ds41_v2_attn_pipe request=43 on=0\n# ds41_v2_attn_pipe request=44 on=1\n");
    p = {}; assert(!p.on && p.seen.empty());
    puts("ds41_v2_attn_pipe: default off, interleaved slots, bounded once-per-request on/off logs, reinit PASS");
}
