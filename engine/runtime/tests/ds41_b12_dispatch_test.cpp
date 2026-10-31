#include "../ds41_b12_dispatch.h"
#include <cstdio>

int main() {
    struct ParseCase { const char* text; int want; } parse_cases[] = {
        {nullptr,0},{"0",0},{"1",1},{"",-1},{"2",-1},{"true",-1},{"01",-1}
    };
    for (const auto& c : parse_cases) {
        if (ds41::b12::parse(c.text) != c.want) return 1;
    }
    // Ship decode budget MX=48, W=4096; all rows at/below MX or beyond W stay old.
    struct RouteCase { bool enabled, draft; int rows, cap; bool want; } route_cases[] = {
        {false,false,4096,4096,false}, {true,true,4096,4096,false},
        {true,false,1,4096,false}, {true,false,48,4096,false},
        {true,false,49,4096,true}, {true,false,128,4096,true},
        {true,false,512,4096,true}, {true,false,2048,4096,true},
        {true,false,4046,4096,true}, {true,false,4096,4096,true},
        {true,false,4097,4096,false}, {true,false,49,48,false}
    };
    for (const auto& c : route_cases) {
        if (ds41::b12::on(c.enabled,c.draft,c.rows,48,c.cap) != c.want) return 2;
    }
    // The boundary must also hold for the alternate 32-row decode build.
    if (ds41::b12::on(true,false,32,32,4096) || !ds41::b12::on(true,false,33,32,4096)) return 3;
    std::printf("B12_DISPATCH_CPU_PASS parse=%zu route=%zu alt_mx=2\n",
                sizeof(parse_cases)/sizeof(parse_cases[0]),sizeof(route_cases)/sizeof(route_cases[0]));
    return 0;
}
