// Request grammars, host side (no XGrammar): the gram header of an S2 request (s2/sessions.h parse_grammar) and the row
// of each session that samples its next reply token in a step (s2/dispatch.h grammar_row0), which the masks key on.
#include "s2/dispatch.h"
#include <cstdio>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static bool refused(const std::string& v) {
    try { parse_grammar(v); return false; } catch (const std::exception&) { return true; }
}
// a session's rows: positions pos0 .. pos0 + n - 1, tokens 100 + position
static void add(s2::Step& s, int q, int pos0, int n) {
    s.sess.push_back(q); s.row0.push_back(s.M()); s.rows.push_back(n); s.ctx.push_back(pos0 + n);
    for (int i = 0; i < n; ++i) s.r.push_back({q, 100 + pos0 + i, pos0 + i, pos0 + i + 1, -1});
}

int main() {
    // the header: kind, strict, the arming token, the schema's bytes as hex
    auto g = parse_grammar("2:1:154842:" + std::string("7b7d"));
    CHECK(g.kind == g6rt::GrammarSpec::SCHEMA && g.strict && g.arm == 154842 && g.schema == "{}");
    g = parse_grammar("1:0:-1:");
    CHECK(g.kind == g6rt::GrammarSpec::JSON && !g.strict && g.arm == -1 && g.schema.empty());
    for (const char* bad : {"3:0:-1:", "2:2:-1:7b7d", "2:0:-1:7b7", "2:0:-1:zz", "2:0:-1:", "1:0:-1:7b7d", "2:0:7b7d"})
        CHECK(refused(bad));

    // a 40-token prompt's tail: rows 36..39; the last samples the first reply token
    s2::Step tail; add(tail, 0, 36, 4);
    CHECK(grammar_row0(tail, 0, 40 + 0 - 1) == 3);
    // an earlier chunk of the same prompt: nothing in it samples a reply
    s2::Step early; add(early, 0, 32, 4);
    CHECK(grammar_row0(early, 0, 39) == -1);
    // an MTP verify after 5 reply tokens: row 0's input is the last of them (position 44), rows 1-2 hold drafts
    s2::Step verify; add(verify, 0, 44, 3);
    CHECK(grammar_row0(verify, 0, 40 + 5 - 1) == 0);
    // inserted tokens run as the session's own rows: the last one samples
    s2::Step forced; add(forced, 0, 44, 3);
    CHECK(grammar_row0(forced, 0, 40 + 7 - 1) == 2);
    // two sessions in one round: each finds its own row
    s2::Step two; add(two, 0, 44, 3); add(two, 1, 10, 3);
    CHECK(grammar_row0(two, 0, 44) == 0 && grammar_row0(two, 1, 10) == 3 && grammar_row0(two, 1, 44) == -1);

    std::printf(fails ? "grammar_request_test: %d FAIL\n" : "grammar_request_test: PASS\n", fails);
    return fails != 0;
}
