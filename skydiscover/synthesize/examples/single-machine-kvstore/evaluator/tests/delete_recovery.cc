// Delete durability: a delete acknowledged at the last Checkpoint() stays deleted after a crash.
// Catches a store that persists live keys but not tombstones, so deleted keys come back on recovery.
//
// Two processes. Child: write N keys, Checkpoint(), delete half, Checkpoint() again, raise(SIGKILL).
// Parent: recover, check the deleted keys are absent and the rest byte-exact.
#include "kvstore_interface.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <string>
#include <unistd.h>
#include <sys/wait.h>

static const char* kPath = "/tmp/check_delete_recovery";
static const int kN = 40000;  // delete the even keys -> half live, half deleted

static void make_value(GenValue& v, uint64_t key) {
    uint32_t sz = (key % 2 == 0) ? 64u : 4096u;
    v.size = sz;
    for (uint32_t i = 0; i < sz; ++i)
        v.data[i] = (uint8_t)((key * 2862933555777941757ull + i * 7) >> 11);
}
static bool is_deleted(uint64_t k) { return (k % 2) == 0; }

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
        store->Checkpoint();                                   // first checkpoint: all keys live
        for (uint64_t k = 0; k < kN; ++k) if (is_deleted(k)) store->Delete(k);
        store->Checkpoint();                                   // second checkpoint: deletes must persist
        raise(SIGKILL);
        _exit(99);
    }

    int status = 0;
    waitpid(pid, &status, 0);
    if (!(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL)) {
        printf("DELETE_RECOVERY child did not die by SIGKILL (status=%d) -> FAIL\n", status);
        return 1;
    }

    IKVStore* store = create_kvstore();
    store->InitExtended(1 << 20, 0, 128ull * 1024 * 1024, kPath);
    store->StartSession();
    store->Refresh();

    uint64_t live_ok = 0, live_missing = 0, live_wrong = 0, del_absent = 0, del_phantom = 0;
    GenValue out, exp;
    for (uint64_t k = 0; k < kN; ++k) {
        bool found = store->Read(k, out);
        if (is_deleted(k)) { if (found) ++del_phantom; else ++del_absent; }
        else {
            make_value(exp, k);
            if (!found) ++live_missing;
            else if (out.size != exp.size || memcmp(out.data, exp.data, exp.size) != 0) ++live_wrong;
            else ++live_ok;
        }
    }
    store->StopSession();
    system(("rm -rf " + q + " 2>/dev/null").c_str());

    bool pass = (live_missing == 0 && live_wrong == 0 && del_phantom == 0 &&
                 live_ok == (uint64_t)(kN / 2) && del_absent == (uint64_t)(kN / 2));
    printf("DELETE_RECOVERY live_ok=%llu live_missing=%llu live_wrong=%llu del_absent=%llu del_phantom=%llu -> %s\n",
           (unsigned long long)live_ok, (unsigned long long)live_missing, (unsigned long long)live_wrong,
           (unsigned long long)del_absent, (unsigned long long)del_phantom, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
