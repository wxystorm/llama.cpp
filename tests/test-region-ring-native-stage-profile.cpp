#include "llama-region-ring-native-stage-profile.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "GCNA_NATIVE_TEST_FAIL line=%d\n", __LINE__); \
    return 1; \
} } while (0)

int main() {
    using namespace llama_ring_native_stage;
    const std::string file = "region-ring-native-stage-profile-test.csv";
    auto write = [&](bool failed_npu) {
        std::ofstream out(file, std::ios::trunc);
        out << "backend,layers,chunk,kv_tokens,compute_median_ms,"
               "prepare_median_ms,weight_bytes,kv_bytes,"
               "graph_workspace_bytes,parity,source,"
               "raw_weight_bytes,same_bank_refill_median_ms\n";
        const char * source = "NATIVE_SINGLE_BACKEND_GGUF_PREFIX";
        out << "G,2,64,512,4,0.1,1000,100,100,NOT_REQUESTED,"
            << source << ",1000,0\n";
        out << "C,2,64,512,12,0.4,1000,100,100,NOT_REQUESTED,"
            << source << ",1000,0\n";
        out << "N,1,32,512,8,0.3,2500,100,100,PASS,"
            << source << ",2100,3\n";
        out << "N,1,64,512,12,0.5,2500,100,100,"
            << (failed_npu ? "FAIL" : "PASS") << ','
            << source << ",2100,4\n";
        out << "N,1,64,1024,14,0.6,2500,100,100,PASS,"
            << source << ",2100,4\n";
        out << "A,2,64,512,17,1.2,1700,100,100,NOT_REQUESTED,"
            << source << ",1700,0\n";
    };
    write(false);
    profile p;
    std::string reason;
    CHECK(p.load(file, reason));
    CHECK(p.size() == 6);
    CHECK(p.has_shape('N', 1));
    CHECK(!p.has_shape('N', 4));
    CHECK(p.has_shape('A', 2));
    double compute = 0;
    CHECK(p.estimate('N', 1, 48, 512, phase::COMPUTE, compute));
    CHECK(std::fabs(compute - 10.0) < 1e-9);
    CHECK(p.estimate('N', 1, 64, 768, phase::COMPUTE, compute));
    CHECK(std::fabs(compute - 13.0) < 1e-9);
    CHECK(!p.estimate('N', 1, 96, 512, phase::COMPUTE, compute));
    CHECK(p.estimate('N', 4, 64, 512, phase::COMPUTE, compute));
    CHECK(std::fabs(compute - 48.0) < 1e-9);
    CHECK(!p.estimate('N', 33, 64, 512, phase::COMPUTE, compute));
    CHECK(!p.estimate('N', 4, 96, 512, phase::COMPUTE, compute));
    size_t weight = 0;
    CHECK(p.weight_bytes('N', 1, true, weight));
    CHECK(weight == 2100);
    CHECK(p.weight_bytes('N', 1, false, weight));
    CHECK(weight == 2500);
    CHECK(p.weight_bytes('N', 4, false, weight));
    CHECK(weight >= 10500);
    CHECK(p.weight_bytes('N', 4, true, weight));
    CHECK(weight >= 8820);
    CHECK(p.workspace_bytes('N', 4, weight));
    CHECK(weight >= 440);
    // Cheap E/Beam uses measured N1 + an Adreno suffix. This is only a
    // shortlist proxy; the exact DAG still rejects unmeasured A3/A5 shapes.
    CHECK(p.estimate_phone_coarse(
        3, 64, 512, phase::COMPUTE, compute));
    CHECK(std::fabs(compute - 29.0) < 1e-9);
    CHECK(p.estimate_phone_coarse(
        5, 64, 512, phase::COMPUTE, compute));
    CHECK(std::fabs(compute - 46.0) < 1e-9);
    CHECK(p.npu_refill_ms(2100, compute));
    CHECK(std::fabs(compute - 4.0) < 1e-9);
    CHECK(!p.npu_refill_ms(2101, compute));
    write(true);
    CHECK(!p.load(file, reason));
    CHECK(reason == "UNSAFE_OR_DUPLICATE_NATIVE_STAGE");
    std::remove(file.c_str());
    std::puts("GCNA_NATIVE_TEST status=PASS");
    return 0;
}
