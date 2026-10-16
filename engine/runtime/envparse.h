// Strict environment-variable parsing (host only, no CUDA): unset = the default; set = the WHOLE string must be one of
// the allowed decimal values, else the caller dies before any work. (atoi fails open: "", "x" -> 0, "1junk" -> 1, "2.5" -> 2.)
#pragma once
#include <cstdlib>
#include <cstring>
#include <string>

// value of `name` if it is unset (-> dflt) or exactly one of "0".."max" with no sign, space, leading zero or suffix;
// otherwise -1 and *err = a message naming the variable and the bad value
inline int env_choice(const char* name, int max, int dflt, std::string* err)
{
    const char* v = std::getenv(name);
    if (!v) return dflt;
    if (std::strlen(v) == 1 && v[0] >= '0' && v[0] - '0' <= max) return v[0] - '0';
    if (err) *err = std::string(name) + "='" + v + "' is not one of 0.." + std::to_string(max) + " (unset = " + std::to_string(dflt) + ")";
    return -1;
}

// env_choice for a range that does not start at 0 (G6_MOE_TOPK wants 1..TOPK: 0 is not a valid cut). Same contract:
// unset = dflt, else exactly one decimal digit lo..hi, otherwise -1 and *err names the variable and the bad value.
inline int env_digit(const char* name, int lo, int hi, int dflt, std::string* err)
{
    const char* v = std::getenv(name);
    if (!v) return dflt;
    if (std::strlen(v) == 1 && v[0] >= '0' + lo && v[0] <= '0' + hi) return v[0] - '0';
    if (err) *err = std::string(name) + "='" + v + "' is not one of " + std::to_string(lo) + ".." + std::to_string(hi) +
                        " (unset = " + std::to_string(dflt) + ")";
    return -1;
}

// The G6_MOE_TOPK cut (Flash model change, screen arm): routed experts kept per token, renormalized over the kept.
// Boot (g6m4 / m4sim) dies on a bad value before any work, so the static read here can trust the environment; the
// fallback keeps the model default if the variable vanishes between boot and here. Host only. Single-digit TOPK only.
inline int moe_topk_cut(int model_topk)
{
    static const int k = [model_topk]
    {
        std::string err;
        const int v = env_digit("G6_MOE_TOPK", 1, model_topk, model_topk, &err);
        return v > 0 ? v : model_topk;
    }();
    return k;
}
