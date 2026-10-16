// Minimal safetensors reader for the KDA tests (mmap, read-only): tensors by name with dtype and shape.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace st
{
struct Tensor
{
    std::string dtype;
    std::vector<long long> shape;
    const uint8_t* p = nullptr;
    size_t bytes = 0;
    long long numel() const { long long n = 1; for (long long s : shape) n *= s; return n; }
    const float* f32() const { return reinterpret_cast<const float*>(p); }
    const uint16_t* u16() const { return reinterpret_cast<const uint16_t*>(p); }
};

struct File
{
    std::map<std::string, Tensor> t;
    std::map<std::string, std::string> meta;
    const uint8_t* base = nullptr;
    size_t size = 0;

    // --- tiny JSON (objects, arrays, strings, numbers) ---
    const char* s = nullptr;
    void ws() { while (*s == ' ' || *s == '\n' || *s == '\t' || *s == '\r') ++s; }
    std::string str()
    {
        std::string o;
        ++s;
        while (*s != '"')
        {
            if (*s == '\\') { ++s; o += *s == 'n' ? '\n' : *s; ++s; continue; }
            o += *s++;
        }
        ++s;
        return o;
    }
    void skip()
    {
        ws();
        if (*s == '"') { str(); return; }
        if (*s == '{' || *s == '[')
        {
            const char close = *s == '{' ? '}' : ']';
            ++s;
            ws();
            while (*s != close)
            {
                skip();
                ws();
                if (*s == ':') { ++s; skip(); ws(); }
                if (*s == ',') { ++s; ws(); }
            }
            ++s;
            return;
        }
        while (*s && *s != ',' && *s != '}' && *s != ']') ++s;
    }
    std::vector<long long> nums()
    {
        std::vector<long long> v;
        ++s;
        ws();
        while (*s != ']')
        {
            v.push_back(strtoll(s, const_cast<char**>(&s), 10));
            ws();
            if (*s == ',') { ++s; ws(); }
        }
        ++s;
        return v;
    }

    bool open(const std::string& path)
    {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat sb;
        fstat(fd, &sb);
        size = sb.st_size;
        base = static_cast<const uint8_t*>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
        ::close(fd);
        if (base == MAP_FAILED) return false;
        uint64_t n;
        memcpy(&n, base, 8);
        std::string hdr(reinterpret_cast<const char*>(base) + 8, n);
        const uint8_t* data = base + 8 + n;
        s = hdr.c_str();
        ws(); ++s; ws();
        while (*s != '}')
        {
            const std::string key = str();
            ws(); ++s; ws();   // ':'
            if (key == "__metadata__")
            {
                ++s; ws();
                while (*s != '}')
                {
                    const std::string k = str();
                    ws(); ++s; ws();
                    meta[k] = str();
                    ws();
                    if (*s == ',') { ++s; ws(); }
                }
                ++s;
            }
            else
            {
                Tensor x;
                std::vector<long long> off;
                ++s; ws();
                while (*s != '}')
                {
                    const std::string f = str();
                    ws(); ++s; ws();
                    if (f == "dtype") x.dtype = str();
                    else if (f == "shape") x.shape = nums();
                    else if (f == "data_offsets") off = nums();
                    else skip();
                    ws();
                    if (*s == ',') { ++s; ws(); }
                }
                ++s;
                x.p = data + off[0];
                x.bytes = off[1] - off[0];
                t[key] = x;
            }
            ws();
            if (*s == ',') { ++s; ws(); }
        }
        return true;
    }
    const Tensor& operator[](const std::string& k) const
    {
        auto it = t.find(k);
        if (it == t.end()) { fprintf(stderr, "missing tensor %s\n", k.c_str()); exit(2); }
        return it->second;
    }
    bool has(const std::string& k) const { return t.count(k) != 0; }
};
}   // namespace st
