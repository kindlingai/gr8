// Native GLP (GGUF) steering input tests: conformance gates, manifest
// synthesis, determinism, and parity with the schema-v1 directory loader.
// Self-contained: fixtures are synthetic GLP GGUF files written in-test.
#include "../steering_glp.h"
#include <filesystem>
#include <iostream>
#include <unistd.h>

namespace {

using namespace g6rt::steering;

int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    std::cerr << "FAIL " << __LINE__ << ": " << msg << "\n"; ++failures; } } while (0)

const ExpectedModel kModel{"glm-5.3", "rev123",
    std::string(64, 'a'), "tok-v1", std::string(64, 'b'), 8, 78};

std::string tmpdir() {
    const auto dir = std::filesystem::temp_directory_path() /
        ("steering_glp_test_" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir);
    return dir.string();
}

// ------------------------------------------------- fixture GGUF writer
struct Fixture {
    std::map<std::string, std::string> strings;
    std::map<std::string, uint32_t> uints;
    std::map<std::string, float> floats;
    std::vector<std::pair<std::string, std::vector<float>>> tensors; // in write order
};

void put_u64(std::string& b, uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(char(v >> (8 * i)));
}
void put_u32(std::string& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(char(v >> (8 * i)));
}
void put_str(std::string& b, const std::string& s) { put_u64(b, s.size()); b += s; }

std::string write_gguf(const std::string& path, const Fixture& f) {
    std::string b = "GGUF";
    put_u32(b, 3);
    put_u64(b, f.tensors.size());
    put_u64(b, f.strings.size() + f.uints.size() + f.floats.size());
    for (const auto& [k, v] : f.strings) { put_str(b, k); put_u32(b, 8); put_str(b, v); }
    for (const auto& [k, v] : f.uints) { put_str(b, k); put_u32(b, 4); put_u32(b, v); }
    for (const auto& [k, v] : f.floats) {
        put_str(b, k); put_u32(b, 6);
        uint32_t bits; std::memcpy(&bits, &v, 4); put_u32(b, bits);
    }
    uint64_t offset = 0;
    for (const auto& [name, data] : f.tensors) {
        put_str(b, name); put_u32(b, 1); put_u64(b, data.size()); put_u32(b, 0);
        put_u64(b, offset); offset += data.size() * 4;
    }
    while (b.size() % 32) b.push_back('\0');
    for (const auto& [name, data] : f.tensors)
        for (float v : data) { uint32_t bits; std::memcpy(&bits, &v, 4); put_u32(b, bits); }
    std::ofstream o(path, std::ios::binary);
    o.write(b.data(), std::streamsize(b.size()));
    return b;
}

std::vector<float> unit_vec(uint32_t width, size_t at) {
    std::vector<float> v(width, 0.0f);
    v[at % width] = 1.0f;
    return v;
}

std::string tensor_bytes(const std::vector<std::pair<std::string, std::vector<float>>>& ts,
                         const std::vector<std::string>& order) {
    std::string out;
    for (const auto& name : order) {
        const auto it = std::find_if(ts.begin(), ts.end(),
                                     [&](const auto& p) { return p.first == name; });
        for (float v : it->second) { uint32_t bits; std::memcpy(&bits, &v, 4); put_u32(out, bits); }
    }
    return out;
}

Fixture good_glp() {
    Fixture f;
    f.strings["general.name"] = "glp.test-vector";
    f.strings["glp.mode"] = "project";
    f.strings["glp.hook_point"] = "residual_stream_post_layer";
    f.strings["glp.derived_at"] = "residual_stream_post_layer";
    f.strings["glp.base_revision"] = "rev123";
    f.strings["glp.generator"] = "captain-vector 0.5.0";
    f.floats["glp.alpha_default"] = 1.0f;
    f.uints["glp.spec_version"] = 1;
    f.uints["glp.rank"] = 1;
    f.tensors = {{"direction.1", unit_vec(8, 0)}, {"direction.2", unit_vec(8, 1)}};
    f.strings["glp.content_sha256"] =
        digest_hex(g6rt::steering::sha256(tensor_bytes(f.tensors, {"direction.1", "direction.2"}).data(),
                                          2 * 8 * 4));
    return f;
}

Profile load_ok(const Fixture& f, const std::string& dir, const std::string& name = "vec.gguf") {
    const std::string path = dir + "/" + name;
    write_gguf(path, f);
    return load_profile_gguf(path, kModel);
}

void expect_fail(const Fixture& f, const std::string& dir, const std::string& want, int& n) {
    const std::string path = dir + "/bad" + std::to_string(n) + ".gguf";
    write_gguf(path, f);
    try {
        (void)load_profile_gguf(path, kModel);
        CHECK(false, "expected failure: " << want);
    } catch (const std::exception& e) {
        CHECK(std::string(e.what()).find(want) != std::string::npos,
              "failure mentions '" << want << "', got: " << e.what());
    }
}

} // namespace

int main() {
    const std::string dir = tmpdir();
    int badn = 0;

    // 1. valid file: layers, alphas, digests, determinism, content
    const auto p1 = load_ok(good_glp(), dir);
    const auto p2 = load_ok(good_glp(), dir, "vec2.gguf");
    CHECK((p1.layers == std::vector<uint32_t>{1, 2}), "layers parsed");
    CHECK(p1.alphas.size() == 2 && p1.alphas[0] == 1.0f && p1.alphas[1] == 1.0f, "alphas parsed");
    CHECK(p1.scope == Scope::Full, "scope maps to full");
    CHECK(p1.directions.size() == 16, "directions decoded");
    CHECK(p1.directions[0] == 1.0f && p1.directions[1] == 0.0f && p1.directions[9] == 1.0f,
          "direction values");
    CHECK(p1.manifest_json == p2.manifest_json, "manifest synthesis deterministic");
    CHECK(p1.manifest_digest == p2.manifest_digest, "identity deterministic");
    CHECK(p1.schedule_digest == schedule_hash(Scope::Full, {1, 2}, {1.0f, 1.0f}), "schedule digest");
    CHECK(p1.manifest_json.find("\"extraction_tool\":\"glp:captain-vector 0.5.0\"") != std::string::npos,
          "provenance carries GLP generator");

    // 2. layer_scales multiply alpha_default
    {
        auto f = good_glp();
        f.uints["glp.spec_version"] = 2;
        f.strings["glp.layer_scales"] = "2:2.5";
        const auto p = load_ok(f, dir, "scaled.gguf");
        CHECK(p.alphas.size() == 2 && p.alphas[0] == 1.0f && p.alphas[1] == 2.5f,
              "layer_scales applied");
    }

    // 3. parity with the directory loader on the same content
    {
        const std::string pd = dir + "/prof";
        std::filesystem::create_directories(pd);
        std::ofstream m(pd + "/manifest.json");
        m << "{\"schema_version\":1,\"profile_name\":\"glp.test-vector\",\"model\":{"
             "\"name\":\"glm-5.3\",\"revision\":\"rev123\",\"weight_set\":\"" << std::string(64, 'a') <<
             "\",\"tokenizer_revision\":\"tok-v1\",\"chat_template_sha256\":\"" << std::string(64, 'b') <<
             "\",\"hidden_size\":8,\"layer_count\":78},\"scope\":\"full\",\"layers\":[1,2],"
             "\"alphas\":[1.0,1.0],\"directions\":{\"file\":\"directions.f32\",\"dtype\":\"float32-le\","
             "\"byte_count\":64,\"sha256\":\"" <<
             digest_hex(g6rt::steering::sha256(tensor_bytes(good_glp().tensors, {"direction.1", "direction.2"}).data(), 64)) <<
             "\"},\"provenance\":{\"extraction_tool\":\"t\",\"extraction_commit\":\"" <<
             std::string(64, 'c') << "\",\"dataset_manifests\":{\"train\":\"" << std::string(64, 'c') <<
             "\",\"validation\":\"" << std::string(64, 'c') << "\",\"final\":\"" << std::string(64, 'c') <<
             "\"}}}" << std::endl;
        const auto payload = tensor_bytes(good_glp().tensors, {"direction.1", "direction.2"});
        std::ofstream d(pd + "/directions.f32", std::ios::binary);
        d.write(payload.data(), std::streamsize(payload.size()));
        d.close();
        const auto dirp = load_profile(pd, kModel);
        CHECK(dirp.schedule_digest == p1.schedule_digest, "schedule digest parity");
        CHECK(dirp.directions == p1.directions, "direction bytes parity");
        CHECK(dirp.scope == p1.scope, "scope parity");
    }

    // 4. load_profile_any dispatches on magic
    {
        const auto any_gguf = load_profile_any(dir + "/vec.gguf", kModel);
        CHECK(any_gguf.manifest_digest == p1.manifest_digest, "dispatch loads GGUF");
        const auto any_dir = load_profile_any(dir + "/prof", kModel);
        CHECK(any_dir.schedule_digest == p1.schedule_digest, "dispatch loads directory");
    }

    // 5. conformance gates
    { auto f = good_glp(); f.strings["glp.mode"] = "add"; expect_fail(f, dir, "project", ++badn); }
    { auto f = good_glp(); f.strings.erase("glp.mode"); expect_fail(f, dir, "missing", ++badn); }
    { auto f = good_glp(); f.strings["glp.mode"] = "multiply"; expect_fail(f, dir, "project", ++badn); }
    { auto f = good_glp(); f.uints["glp.spec_version"] = 4; expect_fail(f, dir, "spec_version", ++badn); }
    { auto f = good_glp(); f.strings["glp.hook_point"] = "attn_out_pre_residual";
      expect_fail(f, dir, "apply site", ++badn); }
    { auto f = good_glp(); f.strings["glp.hook_point"] = "hc_fold_pre_engram";
      f.uints["glp.spec_version"] = 3; expect_fail(f, dir, "apply site", ++badn); }
    { auto f = good_glp(); f.uints["glp.rank"] = 2;
      f.tensors.push_back({"direction.1.1", unit_vec(8, 2)});
      expect_fail(f, dir, "rank", ++badn); }
    { auto f = good_glp(); f.tensors.push_back({"direction.0", unit_vec(8, 3)});
      f.strings["glp.content_sha256"] = std::string(64, '0');
      expect_fail(f, dir, "direction.0", ++badn); }
    { auto f = good_glp(); f.tensors[1].second = unit_vec(8, 0); // duplicate direction, fine
      f.tensors.push_back({"direction.3", std::vector<float>(4, 0.0f)}); // wrong width, bad norm anyway
      expect_fail(f, dir, "", ++badn); } // any of width/layer/digest gate fires
    { auto f = good_glp(); f.tensors[1].second = std::vector<float>(8, 0.5f); // norm > 1.001
      expect_fail(f, dir, "", ++badn); }
    { auto f = good_glp(); f.floats["glp.alpha_default"] = 5.0f; expect_fail(f, dir, "[0, 4]", ++badn); }
    { auto f = good_glp(); f.uints["glp.spec_version"] = 2; f.strings["glp.layer_scales"] = "2:5.0";
      expect_fail(f, dir, "[0, 4]", ++badn); }
    { auto f = good_glp(); f.strings["glp.base_revision"] = "other-rev";
      expect_fail(f, dir, "revision", ++badn); }
    { auto f = good_glp(); f.strings["glp.content_sha256"] = std::string(64, 'f');
      expect_fail(f, dir, "content_sha256", ++badn); }
    { auto f = good_glp(); f.strings.erase("glp.content_sha256");
      expect_fail(f, dir, "content_sha256", ++badn); }
    { auto f = good_glp(); f.uints["glp.rank"] = 1; f.strings["glp.dir_scales"] = "1.0";
      expect_fail(f, dir, "dir_scales", ++badn); }

    // 6. truncated/corrupt containers
    {
        const std::string path = dir + "/trunc.gguf";
        { std::ofstream o(path, std::ios::binary); o << "GGUF\x03\0\0\0"; }
        try { (void)load_profile_gguf(path, kModel); CHECK(false, "truncated refused"); }
        catch (const std::exception&) {}
        const std::string notgg = dir + "/plain.bin";
        { std::ofstream o(notgg, std::ios::binary); o << "NOPE"; }
        try { (void)load_profile_gguf(notgg, kModel); CHECK(false, "non-GGUF refused"); }
        catch (const std::exception& e) {
            CHECK(std::string(e.what()).find("not a GGUF") != std::string::npos, "magic error");
        }
    }

    std::filesystem::remove_all(dir);
    if (failures) { std::cerr << failures << " failure(s)\n"; return 1; }
    std::cout << "steering_glp_test: OK\n";
    return 0;
}
