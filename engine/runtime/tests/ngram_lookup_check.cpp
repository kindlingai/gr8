// ngram_lookup_check TABLE KMAX: ngram_lookup.h's chain for each context on stdin (one line of token ids), printed as
// "tok:conf tok:conf ..." (conf to 6 places). tools/ngram/tests compare it with ngram.py's engine_chain on the same
// exported file (NGRAM_LOOKUP_CHECK=<this binary>).
#include "ngram_lookup.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>

int main(int argc, char** argv)
{
    if (argc != 3) { std::cerr << "usage: ngram_lookup_check TABLE KMAX\n"; return 2; }
    std::ifstream f(argv[1]);
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    g6rt::NgramTable t;
    std::string err;
    if (!t.parse(text, &err)) { std::cerr << err << "\n"; return 1; }
    const int kmax = atoi(argv[2]);
    std::string line;
    while (std::getline(std::cin, line))
    {
        std::istringstream ls(line);
        std::vector<int> ctx;
        int v;
        while (ls >> v) ctx.push_back(v);
        std::vector<int> out(kmax + 1);
        std::vector<double> c(kmax + 1);
        const int k = t.chain(ctx.data(), (int)ctx.size(), kmax, out.data(), c.data());
        for (int j = 1; j <= k; ++j) printf("%s%d:%.6f", j > 1 ? " " : "", out[j], c[j]);
        printf("\n");
    }
    return 0;
}
