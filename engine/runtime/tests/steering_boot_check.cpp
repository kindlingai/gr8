// CPU bridge for local boot preparation and the actual rank agreement receipt.
#include "../steering_boot.h"
#include "../s2/bootstrap.h"
#include <iostream>
int main(int argc,char** argv) {
    using namespace g6rt::steering;
    if(argc!=6) return 2;
    try {
        const auto j=g6v::Json::parse(read_file(argv[2]));
        ExpectedModel e{j.at("name").str(),j.at("revision").str(),j.at("weight_set").str(),
            j.at("tokenizer_revision").str(),j.at("chat_template_sha256").str(),u32_field(j,"hidden_size"),u32_field(j,"layer_count")};
        std::vector<int> layers; std::stringstream stream(argv[4]); std::string value;
        while(std::getline(stream,value,',')) layers.push_back(std::stoi(value));
        const auto boot=prepare_boot(std::string(argv[1])=="-" ? "":argv[1],e,argv[3],layers,std::string(argv[5])=="-" ? "":argv[5]);
        g6s2::BootstrapConfig c; c.ranks=4; c.directory="/tmp/unit"; c.runtime_signature="unit"; c.scheduler.steering=boot.config;
        std::cout<<c.agreement()<<"\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
}
