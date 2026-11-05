// s2_log.h: the text form of the planner's round log (GR8_S2_REPLAY_LOG, s2_batch.h Batch::record). Host code only.
//
// One record a line: a tag, then the fields of the values it carries, space-separated. Integers are decimal, floating
// point values are C99 hex floats (%a), so a value reads back to the same bits. A vector is its length and then its
// elements. A struct is the fields its `fields(V&, T&)` lists, in order; the same list writes and reads it, so the two
// directions cannot drift apart. s2_batch.h defines `fields` next to each struct it logs.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace s2log {

// appends the fields of the values passed to it to one line
struct Out
{
    std::string line;
    template <class T> Out& operator()(const T& x)
    {
        using U = std::remove_cv_t<T>;
        if constexpr (std::is_same_v<U, bool>) put(x ? "1" : "0");
        else if constexpr (std::is_unsigned_v<U>) put(std::to_string((unsigned long long)x));
        else if constexpr (std::is_integral_v<U> || std::is_enum_v<U>) put(std::to_string((long long)x));
        else if constexpr (std::is_floating_point_v<U>)
        {
            char b[48];
            snprintf(b, sizeof b, "%a", (double)x);
            put(b);
        }
        else if constexpr (std::is_array_v<U>) { for (const auto& e : x) (*this)(e); }
        else fields(*this, const_cast<U&>(x));
        return *this;
    }
    template <class T> Out& operator()(const std::vector<T>& v)
    {
        (*this)((long long)v.size());
        for (const T& e : v) (*this)(e);
        return *this;
    }
    void put(const std::string& s) { line += ' '; line += s; }
};

// reads the fields of one line back in the order Out wrote them
struct In
{
    const char* p;
    explicit In(const char* s) : p(s) {}
    std::string word()
    {
        while (*p == ' ') ++p;
        const char* a = p;
        while (*p && *p != ' ' && *p != '\n') ++p;
        if (a == p) throw std::runtime_error("s2log: short record");
        return std::string(a, p);
    }
    template <class T> In& operator()(T& x)
    {
        using U = std::remove_cv_t<T>;
        if constexpr (std::is_same_v<U, bool>) x = word() == "1";
        else if constexpr (std::is_enum_v<U>) x = (U)std::stoll(word());
        else if constexpr (std::is_unsigned_v<U>) x = (U)std::stoull(word());
        else if constexpr (std::is_integral_v<U>) x = (U)std::stoll(word());
        else if constexpr (std::is_floating_point_v<U>)
        {
            const std::string w = word();
            char* end = nullptr;
            x = (U)std::strtod(w.c_str(), &end);
            if (*end) throw std::runtime_error("s2log: not a number: " + w);
        }
        else if constexpr (std::is_array_v<U>) { for (auto& e : x) (*this)(e); }
        else fields(*this, x);
        return *this;
    }
    template <class T> In& operator()(std::vector<T>& v)
    {
        long long n = 0;
        (*this)(n);
        if (n < 0 || n > (1LL << 28)) throw std::runtime_error("s2log: bad vector length");
        v.assign((size_t)n, T());
        for (T& e : v) (*this)(e);
        return *this;
    }
};

// one record: the tag and the fields of every value
template <class... T> std::string line(const char* tag, const T&... x)
{
    Out o;
    o.line = tag;
    (o(x), ...);
    return o.line;
}

}   // namespace s2log
