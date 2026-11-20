// Synthetic NVMe/cache/ownership test; no release table or cmap required.
#include "engram/engram_boot.h"
#include <cassert>
#include <random>
#include <fstream>
using namespace ds41::engram;

int main(int argc, char** argv)
{
    assert(argc == 2);
    const std::string dir = argv[1];
    HashParams hp{};
    for (int e=0;e<kNE;++e) for (int j=0;j<kIds;++j) {
        hp.prime[e][j]=31+j+e;
        hp.offset[e][j]=j ? hp.offset[e][j-1]+hp.prime[e][j-1] : 0;
    }
    // Actual production boundaries, both layers: every ID interval has exactly one owner.
    for (int nr : {1,2,3,4,5,6,8,24,25}) for (int e=0;e<kNE;++e) {
        int64_t total=0;
        for (int rank=0;rank<nr;++rank) total+=ColShard::make(kHashParams,e,rank,nr).rows;
        assert(total==kRows[e]);
        for (int j=0;j<kIds;++j) for (auto id : {kHashParams.offset[e][j],kHashParams.offset[e][j]+kHashParams.prime[e][j]-1}) {
            int owned=0;
            for (int rank=0;rank<nr;++rank) {
                const auto c=ColShard::make(kHashParams,e,rank,nr);
                assert(c.at(id)==c.at(j,id));
                owned+=c.at(id)>=0;
                assert((c.at(id)>=0)==(j%nr==rank));
            }
            assert(owned==1);
        }
    }
    // Real file rows with arbitrary bytes, across both layers and split-file run boundaries.
    std::vector<TableFile> full;
    for (int e=0;e<kNE;++e) {
        const int n=hp.offset[e][23]+hp.prime[e][23];
        const std::string path=dir+"/table"+std::to_string(e);
        std::vector<uint8_t> bytes(n*kRowBytes);
        for(size_t i=0;i<bytes.size();++i) bytes[i]=(i*17+(i/256)*13+e*91)%256;
        std::ofstream f(path,std::ios::binary);f.write((char*)bytes.data(),bytes.size());f.close();
        full.push_back({path,0,n,{}, {}});
    }
    StoreOpts opts;opts.cache_bytes=4<<20;opts.io_threads=2;opts.direct=false;
    std::mt19937 rng(723);
    for (int rank=0;rank<4;++rank) {
        RowStore col,old;std::string err;std::vector<TableFile> cf,of;
        for(int e=0;e<kNE;++e) {
            const int64_t n=full[e].valid,part=(n+3)/4,first=rank*part,valid=std::min(part,n-first);
            of.push_back({"",first,valid,{{full[e].path,first,valid,uint64_t(first)*kRowBytes}}, {}});
            const auto c=ColShard::make(hp,e,rank,4);
            TableFile f{"",hp.offset[e][rank],c.rows,{},c};
            for(int j=rank;j<kIds;j+=4) {
                const int64_t a=hp.offset[e][j],n=hp.prime[e][j],cut=n/2;
                f.runs.push_back({full[e].path,a,cut,uint64_t(a)*kRowBytes});
                f.runs.push_back({full[e].path,a+cut,n-cut,uint64_t(a+cut)*kRowBytes});
            }
            cf.push_back(f);
        }
        assert(col.open(cf,opts,&err));assert(old.open(of,opts,&err));
        RowStore warmed;assert(warmed.open(cf,opts,&err));
        for(int e=0;e<kNE;++e) {
            std::vector<uint32_t> ids;
            for(int i=0;i<2048;++i)ids.push_back(rng()%full[e].valid);
            ids.push_back(full[e].valid);ids.push_back(0xffffffffu);
            for(int j=0;j<kIds;++j){ids.push_back(hp.offset[e][j]);ids.push_back(hp.offset[e][j]+hp.prime[e][j]-1);}
            std::vector<uint8_t> a(ids.size()*256,0xa5),b(a);
            FetchStats st;
            assert(!col.fetch(e,ids.data(),ids.size(),a.data(),&st));
            assert(!old.fetch(e,ids.data(),ids.size(),b.data()));
            std::ifstream f(full[e].path,std::ios::binary);std::vector<uint8_t> all(full[e].valid*256);f.read((char*)all.data(),all.size());
            for(size_t i=0;i<ids.size();++i)for(int k=0;k<256;++k) {
                assert(a[i*256+k]==(cf[e].columns.at(ids[i])>=0 ? all[size_t(ids[i])*256+k] : 0xa5));
                bool own=int64_t(ids[i])>=of[e].first && int64_t(ids[i])<of[e].first+of[e].valid;
                assert(b[i*256+k]==(own ? all[size_t(ids[i])*256+k] : 0xa5));
                // Combining four ranks produces the same per-cell bytes, not a float tolerance.
            }
            // Lookahead uses this identical fetch with null dst. After warm, zero misses.
            for(int chunk=1;chunk<=3;++chunk) {
                std::vector<uint32_t> future;
                for(int j=0;j<kIds;++j)
                    for(uint32_t k=chunk*hp.prime[e][j]/4;k<(chunk+1)*hp.prime[e][j]/4;++k)
                        future.push_back(hp.offset[e][j]+k);
                // Disjoint upcoming chunks, cold to this store until their warm call.
                assert(!warmed.fetch(e,future.data(),future.size(),nullptr,&st));
                assert(st.miss_req>0);
                std::vector<uint8_t> bytes(future.size()*256,0xa5);
                assert(!warmed.fetch(e,future.data(),future.size(),bytes.data(),&st));
                assert(st.miss_req==0);
            }
        }
        { auto bad=cf;bad[0].runs[0].row0++;RowStore x;assert(!x.open(bad,opts,&err)); }
        { auto bad=cf;bad[0].runs.pop_back();RowStore x;assert(!x.open(bad,opts,&err)); }
    }
    // Sidecar geometry is checked even without a cmap fixture. A valid geometry must
    // reach the missing owned-data file error; malformed ranges/mode must fail before that point.
    for(int rank=0;rank<4;++rank) {
        const std::string prefix=dir+"/rank"+std::to_string(rank);
        auto manifest=[&](bool corrupt) {
            std::string s="{\"format\":\"g6engram-v3-col\",\"rank\":"+std::to_string(rank)+",\"tp\":4,\"tables\":[";
            for(int e=0;e<kNE;++e) {
                const auto c=ColShard::make(kHashParams,e,rank,4);
                s+=(e?",":"")+std::string("{\"layer\":")+std::to_string(kLayerIds[e])+",\"rows\":"+std::to_string(kRows[e])+",\"part\":"+std::to_string(c.rows)+",\"first\":"+std::to_string(c.first[rank])+",\"valid\":"+std::to_string(c.rows)+",\"dim\":256,\"sdim\":8,\"segs\":[";
                for(int j=rank;j<kIds;j+=4) s+=(j==rank?"":",")+std::string("{\"path\":\"unused\",\"row0\":")+std::to_string(c.first[j]+(corrupt&&j==rank?1:0))+",\"rows\":"+std::to_string(c.count[j])+",\"w_off\":0,\"s_off\":0}";
                s+="]}";
            }
            s+="]}";std::ofstream(prefix+".engram.json")<<s;
        };
        manifest(false);Sidecar sc;std::string err;
        assert(!load_sidecar(prefix,rank,4,&sc,&err,true));assert(err.find("unused: column-sharded Engram weight range at 0: cannot open")!=std::string::npos);
        sc={};assert(!load_sidecar(prefix,rank,4,&sc,&err,false));assert(err.find("disagrees")!=std::string::npos);
        manifest(true);sc={};assert(!load_sidecar(prefix,rank,4,&sc,&err,true));assert(err.find("runs are not consecutive")!=std::string::npos);
    }
    puts("PASS col_shard: TP4 partition and random cell bytes, both layers, global tag boundaries, split runs, foreign skip, warm chunks 1-3 zero misses; TP1/2/3/5/6/8/24/25 geometry; sidecar geometry/mode/corruptions (data/cmap absent)");
}
