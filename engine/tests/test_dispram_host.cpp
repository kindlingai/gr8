#include "../dispram_client.h"
#include <cassert>
#include <chrono>
#include <functional>
#include <thread>
#include <vector>
#include <sys/wait.h>
using namespace g6dispram;
static int count_open_fds() { int n=0; for (int f=0;f<1024;++f) if (fcntl(f,F_GETFD)!=-1) ++n; return n; }
static std::string temp_path() { char p[]="/tmp/gr8-dispram-test-XXXXXX"; int f=mkstemp(p); assert(f>=0); close(f); unlink(p); return p; }
static void child_test(char* self) {
    for (const char* mode : {"child_default", "child_on", "child_disable"}) {
        pid_t p=fork(); assert(p>=0);
        if (!p) {
            if (!strcmp(mode,"child_default")) {
                unsetenv("GR8_DISPRAM"); unsetenv("G6_DISPRAM"); unsetenv("DISPRAM_DISABLE");
            } else {
                setenv("GR8_DISPRAM","1",1);
                setenv("DISPRAM_DISABLE",!strcmp(mode,"child_disable")?"1":"0",1);
                setenv("DISPRAM_SOCKET","/tmp/dispram-child.sock",1); setenv("GR8_DISPRAM_MIN_BYTES","1983905792",1);
            }
            char* args[]={self,const_cast<char*>(mode),nullptr}; execv(self,args); _exit(99);
        }
        int st=0; assert(waitpid(p,&st,0)==p); assert(WIFEXITED(st) && WEXITSTATUS(st)==0);
    }
}
#ifdef SOCK_SEQPACKET
static std::string recv_request(int fd) { char b[4096]; ssize_t n=recv(fd,b,sizeof(b),0); assert(n>0); return std::string(b,n); }
static void send_reply(int fd,const std::string& text,int nfds=0) {
    std::vector<int> fds;
    for (int i=0;i<nfds;++i) {
        char p[]="/tmp/gr8-dispram-test-XXXXXX"; int f=mkstemp(p); assert(f>=0); unlink(p);
        assert(write(f,"payload",7)==7); assert(lseek(f,0,SEEK_SET)==0); fds.push_back(f);
    }
    iovec io{const_cast<char*>(text.data()),text.size()}; msghdr m{}; m.msg_iov=&io; m.msg_iovlen=1;
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int)*4)]{};
    if (nfds) {
        m.msg_control=control; m.msg_controllen=CMSG_SPACE(sizeof(int)*nfds);
        cmsghdr* c=CMSG_FIRSTHDR(&m); c->cmsg_level=SOL_SOCKET; c->cmsg_type=SCM_RIGHTS;
        c->cmsg_len=CMSG_LEN(sizeof(int)*nfds); memcpy(CMSG_DATA(c),fds.data(),sizeof(int)*nfds);
    }
    assert(sendmsg(fd,&m,0)==static_cast<ssize_t>(text.size()));
    for (int f:fds) close(f);
}
static void mock_server(const std::function<void(int)>& server,const std::function<void(Client&)>& run) {
    int before=count_open_fds(); std::string path=temp_path();
    int s=socket(AF_UNIX,SOCK_SEQPACKET,0); assert(s>=0);
    sockaddr_un a{}; a.sun_family=AF_UNIX; memcpy(a.sun_path,path.c_str(),path.size()+1);
    assert(posix_bind(s,reinterpret_cast<sockaddr*>(&a),sizeof(a))==0); assert(listen(s,1)==0);
    std::thread t([&] { int c=accept(s,nullptr,nullptr); assert(c>=0); server(c); close(c); });
    {
        Client c; assert(c.open_conn(path));
        bool found=false;
        for (int f=0;f<1024;++f) {
            sockaddr_un peer{}; socklen_t size=sizeof(peer);
            if (getpeername(f,reinterpret_cast<sockaddr*>(&peer),&size)==0 && peer.sun_family==AF_UNIX &&
                std::string(peer.sun_path)==path) { assert(fcntl(f,F_GETFD)&FD_CLOEXEC); found=true; }
        }
        assert(found);
        run(c); c.close_conn();
    }
    t.join(); close(s); unlink(path.c_str()); assert(count_open_fds()==before);
}
static void payload(Lease& l) { assert(l.fd>=0 && (fcntl(l.fd,F_GETFD)&FD_CLOEXEC)); char b[7]; assert(read(l.fd,b,7)==7); assert(memcmp(b,"payload",7)==0); close(l.fd); l.fd=-1; }
static void sockets() {
    const std::string info="{\"free\":2145386496,\"size\":2145386496,\"base\":0,\"largest\":2145386496}";
    mock_server([&](int fd) {
        assert(recv_request(fd)=="{\"key\":\"kindlingai_1\",\"op\":\"info\"}"); send_reply(fd,info,2);
        for (int i=0;i<3;++i) { assert(json_has_key(recv_request(fd),"size")); send_reply(fd,"{\"ok\":true,\"size\":2097152}",i==1?2:1); }
        char b; assert(recv(fd,&b,1,0)==0); // disconnect reclaims all leases
    },[](Client& c) { Info i; assert(c.query_info(&i) && i.free_bytes==2145386496ULL); for (int n=0;n<3;++n) { Lease l; assert(c.lease(GRANULE,&l)); payload(l); } });
    for (bool exhausted:{false,true}) mock_server([&](int fd) {
        recv_request(fd); send_reply(fd,info);
        recv_request(fd); send_reply(fd,"{\"ok\":false,\"error\":\"insufficient\"}");
        recv_request(fd); send_reply(fd,exhausted?"{\"free\":0,\"size\":2145386496}":"{\"free\":2097152,\"size\":2145386496}");
        if (!exhausted) { recv_request(fd); send_reply(fd,"{\"ok\":true,\"size\":2097152}",1); }
    },[&](Client& c) { Info i; Lease l; assert(c.query_info(&i)); assert(!c.lease(2*GRANULE,&l) && c.is_open()); assert(c.query_info(&i)); assert(i.free_bytes==(exhausted?0:GRANULE)); if (!exhausted) { assert(c.lease(GRANULE,&l)); payload(l); } });
    for (int mode=0;mode<4;++mode) mock_server([&](int fd) {
        recv_request(fd);
        if (mode==0) send_reply(fd,"{\"ok\":trux}",1);
        if (mode==1) send_reply(fd,"{\"ok\":true,\"size\":2097152}");
        if (mode==2) std::this_thread::sleep_for(std::chrono::milliseconds(250));
        // mode 3: disappear before the lease reply
    },[](Client& c) { Lease l; auto start=std::chrono::steady_clock::now(); assert(!c.lease(GRANULE,&l,150)); assert(c.is_open() && c.is_broken()); assert(!c.lease(GRANULE,&l,150)); c.close_conn(); assert(!c.is_open() && !c.is_broken()); assert(std::chrono::steady_clock::now()-start<std::chrono::seconds(1)); });
}
struct MockBackend {
    using Err=int;
    static constexpr Err ok_err() { return 0; }
    static constexpr Err invalid_err() { return 1; }
    Session<MockBackend>* session=nullptr;
    int report=-1;
    bool allow_fallback=false;
    int sync_count=0,unmap_count=0,fallback_count=0;
    void event(char c) { assert(write(report,&c,1)==1); }
    bool granularity(size_t* out) { *out=GRANULE; return true; }
    bool map_span(int fd,const SpanPlan& plan,size_t,int,size_t,uintptr_t* out) {
        close(fd); *out=reinterpret_cast<uintptr_t>(new char[plan.total_bytes]); return true;
    }
    void sync() { ++sync_count; event('S'); }
    void unmap_span(uintptr_t va,const Span&) {
        assert(sync_count==unmap_count+1);
        assert(session->has_open_conn_for_test());
        ++unmap_count; event('U'); event('O'); delete[] reinterpret_cast<char*>(va);
    }
    Err fallback_alloc(void** out,size_t n) {
        ++fallback_count; event('F'); assert(allow_fallback);
        *out=new char[n]; return 0;
    }
    Err fallback_free(void* p) { delete[] static_cast<char*>(p); return 0; }
};
static int session_child(int mode,int report) {
    MockBackend backend; backend.report=report; backend.allow_fallback=mode>=4;
    Session<MockBackend> session(backend); backend.session=&session; session.set_timeout_ms_for_test(100);
    void *a=nullptr,*b=nullptr;
    assert(session.alloc_kv(&a,mode==5?GRANULE:2*GRANULE)==0);
    if (mode==5) {
        assert(!session.has_open_conn_for_test() && session.claimed_bytes()==0);
        assert(backend.fallback_count==1); session.free_kv(a); return 0;
    }
    assert(session.claimed_bytes()==2*GRANULE);
    assert(session.alloc_kv(&b,2*GRANULE)==0);
    assert(mode==4); // All transport loss cases must have exited 75, never continued here.
    assert(session.has_open_conn_for_test() && !session.is_broken_for_test());
    assert(session.claimed_bytes()==2*GRANULE && backend.fallback_count==1);
    session.free_kv(b); assert(session.has_open_conn_for_test());
    session.free_kv(a);
    assert(!session.has_open_conn_for_test() && session.claimed_bytes()==0 && session.mapped_bytes()==0);
    assert(backend.sync_count==1 && backend.unmap_count==1); return 0;
}
static void live_span_transport_tests(char* self) {
    const int before=count_open_fds();
    for (int min_positive=0;min_positive<2;++min_positive) for (int mode=0;mode<(min_positive?4:6);++mode) {
        std::string path=temp_path(); int listen_fd=socket(AF_UNIX,SOCK_SEQPACKET,0); assert(listen_fd>=0);
        sockaddr_un addr{}; addr.sun_family=AF_UNIX; memcpy(addr.sun_path,path.c_str(),path.size()+1);
        assert(posix_bind(listen_fd,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))==0 && listen(listen_fd,1)==0);
        int events[2]; assert(pipe(events)==0);
        char errpath[]="/tmp/gr8-dispram-test-XXXXXX"; int err=mkstemp(errpath); assert(err>=0); unlink(errpath);
        const std::string mode_arg=std::to_string(mode),fd_arg=std::to_string(events[1]);
        pid_t child=fork(); assert(child>=0);
        if (!child) {
            close(listen_fd); close(events[0]); assert(dup2(err,2)==2); close(err);
            setenv("GR8_DISPRAM","1",1); setenv("DISPRAM_DISABLE","0",1);
            if (mode==5) { unsetenv("GR8_DISPRAM"); unsetenv("G6_DISPRAM"); }
            setenv("DISPRAM_SOCKET",path.c_str(),1); setenv("GR8_DISPRAM_MIN_BYTES",min_positive?"6291456":"0",1);
            char arg[]="session";
            char* args[]={self,arg,const_cast<char*>(mode_arg.c_str()),const_cast<char*>(fd_arg.c_str()),nullptr};
            execv(self,args); _exit(99);
        }
        close(events[1]);
        std::thread server([&] {
            if (mode==5) return; // Default-off never opens a daemon connection.
            int fd=accept(listen_fd,nullptr,nullptr); assert(fd>=0);
            recv_request(fd); send_reply(fd,"{\"free\":16777216,\"size\":16777216}");
            recv_request(fd);
            if (mode==5) { send_reply(fd,"{\"ok\":trux}",1); }
            else {
                send_reply(fd,"{\"ok\":true,\"size\":4194304}",1);
                if (mode!=3) {
                    recv_request(fd);
                    if (mode==0) send_reply(fd,"{\"ok\":trux}",1);
                    if (mode==1) send_reply(fd,"{\"ok\":true,\"size\":4194304}");
                    if (mode==2) std::this_thread::sleep_for(std::chrono::milliseconds(250));
                    if (mode==4) {
                        send_reply(fd,"{\"ok\":false,\"error\":\"insufficient\"}");
                        recv_request(fd); send_reply(fd,"{\"free\":0,\"size\":16777216}");
                        char eof; assert(recv(fd,&eof,1,0)==0);
                    }
                }
            }
            close(fd);
        });
        int st=0; assert(waitpid(child,&st,0)==child); server.join();
        if (mode==5) { pollfd pending{listen_fd,POLLIN,0}; assert(poll(&pending,1,0)==0); }
        assert(WIFEXITED(st) && WEXITSTATUS(st)==(mode<4?75:0));
        std::string observed; char buf[1024]; ssize_t n;
        while ((n=read(events[0],buf,sizeof(buf)))>0) observed.append(buf,n);
        assert(observed==(mode<4?"SUO":mode==4?"FSUO":"F"));
        assert(lseek(err,0,SEEK_SET)==0); n=read(err,buf,sizeof(buf)); assert(n>=0);
        if (mode<4) assert(std::string(buf,n).find("g6: dispram shortfall:")!=std::string::npos);
        close(events[0]); close(err); close(listen_fd); unlink(path.c_str());
    }
    assert(count_open_fds()==before);
    puts("PASS Session live-span transport: four failures x zero/positive minimum, sync/unmap/open ordering, no continuation; clean exhaustion and empty fail-open");
}

#endif
int main(int argc,char** argv) {
#ifdef SOCK_SEQPACKET
    if (argc==4 && !strcmp(argv[1],"session")) return session_child(std::atoi(argv[2]),std::atoi(argv[3]));
#endif
    if (argc>1) {
        if (!strcmp(argv[1],"child_default")) {
            assert(!knob::dispram() && !knob::dispram_disable() && !enabled());
        } else if (!strcmp(argv[1],"child_on")) {
            assert(knob::dispram() && !knob::dispram_disable() && enabled());
            assert(socket_path()=="/tmp/dispram-child.sock" && knob::dispram_min_bytes()==1983905792);
        } else {
            assert(!strcmp(argv[1],"child_disable"));
            assert(knob::dispram() && knob::dispram_disable() && !enabled());
        }
        return 0;
    }
    child_test(argv[0]); int before=count_open_fds();
    uint64_t remain=2145386496ULL; size_t claimed=0; bool seam=false;
    for (int layer=0;layer<78;++layer) { auto p=plan_span(32*1024*1024,GRANULE,remain); remain-=p.carve_bytes; claimed+=p.carve_bytes; seam|=p.carve_bytes && p.tail_bytes; }
    assert(remain==0 && claimed==2145386496ULL && seam);
    assert(lcm_gran(2,3)==6 && lcm_gran(0,2)==0 && plan_span(0,GRANULE,0).total_bytes==0);
    uint64_t n; bool ok;
    assert(!json_u64("{\"free\":18446744073709551616}","free",&n));
    assert(!json_u64("{\"free\":-1}","free",&n)); assert(!json_bool("{\"ok\":trueish}","ok",&ok));
    assert(!json_has_key("{\"x\":\"\\\"free\\\":123\"}","free"));
    // Redirect stderr to a portable temporary descriptor to prove absent sockets are silent.
    char p[]="/tmp/gr8-dispram-test-XXXXXX"; int err=mkstemp(p); assert(err>=0); unlink(p);
    int saved=dup(2); assert(saved>=0); assert(dup2(err,2)==2);
    { Client c; assert(!c.open_conn(temp_path(),50)); }
    assert(lseek(err,0,SEEK_END)==0); assert(dup2(saved,2)==2); close(saved); close(err);
#ifdef SOCK_SEQPACKET
    int supported=socket(AF_UNIX,SOCK_SEQPACKET,0);
    if (supported>=0) { close(supported); sockets(); live_span_transport_tests(argv[0]); }
    else { Client c; assert(!c.open_conn(temp_path())); puts("SKIP Unix seqpacket unsupported"); }
#else
    { Client c; assert(!c.open_conn(temp_path())); }
#endif
    assert(count_open_fds()==before);
    puts("PASS dispram host: planner, protocol, descriptor cleanup, retry, timeout, child knobs");
}
