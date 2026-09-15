// Crash durability: data acknowledged at the last Checkpoint() survives a hard process crash.
// Catches a store that persists only on clean shutdown, so kill -9 after Checkpoint() loses data.
//
// Two processes. Child: write N keys, Checkpoint(), raise(SIGKILL), no clean shutdown. Parent: a fresh
// instance on the same path recovers and checks every checkpointed key byte for byte.
#include "kvstore_interface.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <string>
#include <unistd.h>
#include <sys/wait.h>

static const char* kPath = "/tmp/check_crash_durability";
static const int kN = 50000;

static void make_value(GenValue& v, uint64_t key) {
    uint32_t sz = (key % 2 == 0) ? 64u : 4096u;
    v.size = sz;
    for (uint32_t i = 0; i < sz; ++i)
        v.data[i] = (uint8_t)((key * 1099511628211ull + i * 2654435761ull) >> 13);
}

int main() {
    std::string q = std::string("\x27") + kPath + "\x27";
    system(("rm -rf " + q + " 2>/dev/null; mkdir -p " + q).c_str());  // impl-agnostic clean slate

    pid_t pid = fork();
    if (pid == 0) {
        IKVStore* store = create_kvstore();
        store->InitExtended(1 << 20, 0, 128ull * 1024 * 1024, kPath);
        store->StartSession();
        GenValue v;
        for (uint64_t k = 0; k < kN; ++k) { make_value(v, k); store->Upsert(k, v); }
        store->Checkpoint();
        raise(SIGKILL);  // hard crash -- no destructors, no clean StopSession
        _exit(99);       // unreachable
    }

    int status = 0;
    waitpid(pid, &status, 0);
    if (!(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL)) {
        printf("CRASH_DURABILITY child did not die by SIGKILL (status=%d) -> FAIL\n", status);
        return 1;
    }

    IKVStore* store = create_kvstore();
    store->InitExtended(1 << 20, 0, 128ull * 1024 * 1024, kPath);
    store->StartSession();
    store->Refresh();

    uint64_t survived = 0, missing = 0, wrong = 0;
    GenValue out, exp;
    for (uint64_t k = 0; k < kN; ++k) {
        make_value(exp, k);
        if (!store->Read(k, out)) ++missing;
        else if (out.size != exp.size || memcmp(out.data, exp.data, exp.size) != 0) ++wrong;
        else ++survived;
    }
    store->StopSession();
    system(("rm -rf " + q + " 2>/dev/null").c_str());

    bool pass = (survived == (uint64_t)kN && missing == 0 && wrong == 0);
    printf("CRASH_DURABILITY N=%d survived=%llu missing=%llu wrong=%llu -> %s\n",
           kN, (unsigned long long)survived, (unsigned long long)missing,
           (unsigned long long)wrong, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
