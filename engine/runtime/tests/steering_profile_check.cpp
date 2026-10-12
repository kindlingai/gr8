// CPU test bridge: compare Python artifact validation with the actual C++ profile loader.
#include "../steering_profile.h"
#include <cstdio>

int main(int argc, char** argv) {
    using namespace g6rt::steering;
    if (argc != 3) {
        std::fprintf(stderr, "usage: steering_profile_check PROFILE EXPECTED_MODEL_JSON\n");
        return 2;
    }
    try {
        const auto model = g6v::Json::parse(read_file(argv[2], MANIFEST_MAX_BYTES));
        object_known(model, "expected model", {"name", "revision", "weight_set", "tokenizer_revision",
                                               "chat_template_sha256", "hidden_size", "layer_count"});
        const ExpectedModel expected{model.at("name").str(), model.at("revision").str(), model.at("weight_set").str(),
            model.at("tokenizer_revision").str(), model.at("chat_template_sha256").str(),
            u32_field(model, "hidden_size"), u32_field(model, "layer_count")};
        const auto profile = load_profile(argv[1], expected);
        std::printf("%s\n", digest_hex(profile.manifest_digest).c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
