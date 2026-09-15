// Exact bytes: every live key reads back exactly what was written; deleted keys read absent.
// Catches corruption, lost keys, resurrected keys, and values recomputed from the key.
//
// Single-threaded model check against a std::unordered_map. Value bytes are a key-independent random
// stream, so recomputing them from the key fails. A budget far below the data forces a spill to disk.
#include "kvstore_interface.h"
#include <unordered_map>
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
static uint64_t g_rng=0x9E3779B97F4A7C15ull; static uint8_t rngb(){g_rng=g_rng*6364136223846793005ull+1442695040888963407ull; return (uint8_t)((g_rng>>33)&0xff);}
static uint64_t mix(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ull;x^=x>>27;x*=0x94d049bb133111ebull;x^=x>>31;return x;}
int main(int argc,char**argv){
  const size_t BUDGET=64ull<<20;          // deliberately << the ~1GB of data, to force the disk tier to spill
  const size_t BUCKETS=1u<<20, LOG=1ull<<26;   // power-of-2, per the interface contract
  const int N=500000;
  std::string path=argc>1?argv[1]:"/tmp/check_value_equiv";
  if(path.empty()){fprintf(stderr,"empty path\n");return 2;}
  std::string q="\x27"+path+"\x27";       // single-quote the path for the shell
  system(("rm -rf "+q+" 2>/dev/null; mkdir -p "+q).c_str());   // clean slate -> deterministic
  IKVStore* s=create_kvstore();
  s->InitExtended(BUCKETS,LOG,BUDGET,path.c_str());
  s->StartSession();
  std::unordered_map<uint64_t,std::vector<uint8_t>> model; std::vector<uint64_t> tombs;
  auto fill=[&](uint64_t k,uint32_t sz){std::vector<uint8_t> b(sz); for(uint32_t j=0;j<sz;j++) b[j]=rngb(); /*key-independent: catches value-regeneration*/ return b;};
  auto put=[&](uint64_t k,uint32_t sz){auto b=fill(k,sz); GenValue v; v.size=sz; memset(v.data,0,sizeof(v.data)); memcpy(v.data,b.data(),sz); s->Upsert(k,v); model[k]=b;};

  for(int i=0;i<N;i++){uint64_t k=mix((uint64_t)i+1); put(k, (i%1000==0)?4096u:64u);}
  for(int i=0;i<N;i+=7){uint64_t k=mix((uint64_t)i+1); put(k,32);}
  for(int i=0;i<N;i+=11){uint64_t k=mix((uint64_t)i+1); s->Delete(k); model.erase(k); tombs.push_back(k);}
  long live=0,wrong=0,lost=0,phantom=0;
  for(auto&kv:model){live++; GenValue r{}; bool ok=s->Read(kv.first,r);
    if(!ok)lost++; else if(r.size!=kv.second.size()||r.size>GenValue::kMaxSize||memcmp(r.data,kv.second.data(),r.size))wrong++;}
  for(uint64_t k:tombs){GenValue r{}; if(s->Read(k,r))phantom++;}     // deleted must read absent
  s->StopSession(); system(("rm -rf "+q+" 2>/dev/null").c_str());
  long fails=wrong+lost+phantom;
  printf("VALUE_EQUIV_CHECK(1thr) live=%ld wrong=%ld lost=%ld phantom_deleted=%ld -> fails=%ld\n",live,wrong,lost,phantom,fails);
  return fails?1:0;
}
