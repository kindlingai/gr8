// g++ -std=c++17 -O2 capture_tool.cpp -o capture_tool
#include "capture_io.h"
#include <iostream>
int main(int argc,char** argv) {
    try {
        if(argc!=3||(std::string(argv[1])!="inspect"&&std::string(argv[1])!="case"))
            throw std::runtime_error("usage: capture_tool inspect|case FILE.asw");
        auto r=sweep_capture::load(argv[2]);const auto& m=r.metadata;
        if(std::string(argv[1])=="inspect") {
            std::cout<<"{\"metadata\":"<<sweep_capture::metadata_json(m)
                <<",\"is_captured\":"<<(r.is_captured?"true":"false")
                <<",\"source_hash\":"<<sweep_capture::quote(r.source_hash)
                <<",\"within_row_unique_fraction\":"<<r.metrics.within_row_unique_fraction
                <<",\"adjacent_same_slot_overlap_fraction\":"<<r.metrics.adjacent_overlap_fraction
                <<",\"adjacent_pairs\":"<<r.metrics.adjacent_pairs<<"}\n";
        } else {
            sweep_capture::require(m.heads==16&&m.session_form=="implicit_single","current sweep adapter needs 16 heads and implicit single slot");
            for(int row=0;row<m.n;++row)
                sweep_capture::require(r.positions[row]==r.positions[0]+row,"current sweep adapter needs contiguous positions");
            std::cout<<"[{\"id\":\"capture_"<<r.source_hash.substr(0,16)<<"\",\"args\":["
                <<m.n<<","<<r.positions[0]<<","<<m.ratio<<","<<m.floor<<","<<m.wl<<","<<std::max(1,m.key_rows)
                <<",\"--capture\","<<sweep_capture::quote(argv[2])<<"],\"weights\":{\"32K\":0,\"128K\":0},"
                <<"\"input_provenance\":\""<<(r.is_captured?"captured_indices_synthetic_operands":"synthetic_test")
                <<"\",\"decision_rows\":false,\"source_hash\":"<<sweep_capture::quote(r.source_hash)<<"}]\n";
        }
        return 0;
    } catch(const std::exception& e) {std::cerr<<"capture: "<<e.what()<<"\n";return 2;}
}
