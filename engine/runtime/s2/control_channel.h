// Persistent event-only host links, used by a native adapter with connected TCP
// sockets. CPU tests use connected local sockets. No socket activity per round.
#pragma once
#include "control_io.h"
#include "control_protocol.h"
#include <functional>

namespace g6s2 {
// Shared event latch: computation only takes a mutex after before_work says due.
// A caller MUST apply the returned commit, then applied(); exceptions fail epoch.
class EventEndpoint {
protected:
    EventGate gate_;
    std::mutex mu_;std::condition_variable cv_;
    std::optional<Notice> notice_;
    std::optional<Commit> commit_;
    std::optional<Ack> ack_;
    bool failed_=false;
    void install(const Notice& n) {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l,[&]{return !notice_ || failed_;});
        require(!failed_,"control epoch failed");
        gate_.announce(n);notice_=n;commit_.reset();ack_.reset();cv_.notify_all();
    }
    void deliver(const EventCommit& e) {
        std::lock_guard<std::mutex> l(mu_);
        require(!failed_ && notice_ && *notice_==e.notice,"unexpected event commit");match(e);
        commit_=e.commit;gate_.commit(e.notice);
    }
    virtual void send_ack(const Ack&)=0;
public:
    EventEndpoint(uint64_t epoch,unsigned rank):gate_(epoch,rank){}
    virtual ~EventEndpoint()=default;
    virtual void fail() {
        {std::lock_guard<std::mutex> l(mu_);failed_=true;}
        gate_.fail();cv_.notify_all();
    }
    // No digest, lock, allocation or I/O if no event is due. Call before departing
    // a LIVE boundary. Idle engines use wait_notice() instead of departing it.
    template<class Digest> std::optional<Commit> boundary(uint64_t b,Digest&& digest) {
        try {
            if(!gate_.before_work(b))return {};
            auto a=gate_.arrive(b,digest());send_ack(a);gate_.wait_commit();
            std::lock_guard<std::mutex> l(mu_);require(commit_.has_value(),"missing commit payload");return commit_;
        }catch(...){fail();throw;}
    }
    Notice wait_notice() {
        std::unique_lock<std::mutex> l(mu_);cv_.wait(l,[&]{return notice_.has_value() || failed_;});
        require(!failed_,"control epoch failed");return *notice_;
    }
    void applied() {
        std::lock_guard<std::mutex> l(mu_);require(notice_ && commit_ && !failed_,"apply without event commit");
        gate_.applied(*notice_);notice_.reset();commit_.reset();ack_.reset();cv_.notify_all();
    }
};
class WorkerChannel : public EventEndpoint {
    ControlIO& io_;
    void send_ack(const Ack& a) override {io_.send(encode(a));}
public:
    WorkerChannel(uint64_t epoch,unsigned rank,ControlIO& io):EventEndpoint(epoch,rank),io_(io){require(rank>0,"worker rank zero");}
    void fail() override {EventEndpoint::fail();io_.shutdown();}
    void receive_events() { // background control thread, one persistent connection
        try {
            for(;;){auto b=io_.receive(true);Decoder d(b); // idle between events: no timeout (#26)
                if(d.tag==Tag::Notice)install(decode_notice(b));
                else if(d.tag==Tag::Commit)deliver(decode_commit(b));
                else throw std::runtime_error("worker received non-event frame");}
        }catch(...){fail();throw;}
    }
};
class LeaderChannel : public EventEndpoint {
    std::vector<ControlIO*> peers_; // order = ranks 1..NR-1
    void send_ack(const Ack& a) override {std::lock_guard<std::mutex> l(mu_);ack_=a;cv_.notify_all();}
public:
    LeaderChannel(uint64_t epoch,const std::vector<ControlIO*>& peers):EventEndpoint(epoch,0),peers_(peers){require(peers.size()==2||peers.size()==3||peers.size()==5,"expected TP3/TP4/TP6 peers");}
    void fail() override {EventEndpoint::fail();for(auto p:peers_)p->shutdown();}
    // Control thread: announce caller-chosen FUTURE boundary, collect event ACKs,
    // then make one canonical allocator/scheduler commit while compute is paused.
    // Ordinary EOS, decode and prefill steps never call this function.
    void event(const Notice& n,const std::function<Commit(uint64_t)>& make_commit) {
        try {
            install(n);for(auto p:peers_)p->send(encode(n));
            Agreement proof(unsigned(peers_.size()+1),n);
            {std::unique_lock<std::mutex> l(mu_);cv_.wait(l,[&]{return ack_.has_value()||failed_;});require(!failed_,"failed while waiting local ACK");proof.ack(*ack_);}
            for(unsigned i=0;i<peers_.size();++i){auto a=decode_ack(peers_[i]->receive());require(a.rank==i+1,"ACK on wrong rank link");proof.ack(a);}
            EventCommit e{n,make_commit(proof.agreed_hash())};match(e);
            require(e.commit.before_hash==proof.agreed_hash(),"commit not based on all-rank state");
            for(auto p:peers_) { p->send(encode(e)); }
            deliver(e);
            std::unique_lock<std::mutex> l(mu_);cv_.wait(l,[&]{return !notice_||failed_;});require(!failed_,"event apply failed");
        }catch(...){fail();throw;}
    }
};
} // namespace g6s2
