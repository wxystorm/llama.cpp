#include "../ggml/src/ggml-rpc/phone-local-host-handoff.h"

#include <cassert>
#include <vector>

int main() {
    std::vector<float> htp { 1.f, -2.f, 3.5f, 4.f };
    const float * const original_allocation = htp.data();
    std::vector<float> adreno;
    assert(ggml_phone_local_host::move_f32(
        htp, adreno, 4 * sizeof(float)));
    assert(adreno.data() == original_allocation);
    assert(adreno.size() == 4);
    assert(adreno[0] == 1.f && adreno[1] == -2.f &&
           adreno[2] == 3.5f && adreno[3] == 4.f);
    assert(htp.empty());

    std::vector<float> mismatched { 5.f, 6.f };
    std::vector<float> unmodified;
    assert(!ggml_phone_local_host::move_f32(
        mismatched, unmodified, sizeof(float)));
    assert(mismatched.size() == 2 && unmodified.empty());
    assert(!ggml_phone_local_host::move_f32(
        mismatched, unmodified, 3 * sizeof(float)));
    assert(mismatched.size() == 2 && unmodified.empty());

    std::vector<float> busy { 7.f };
    assert(!ggml_phone_local_host::move_f32(
        mismatched, busy, 2 * sizeof(float)));
    assert(mismatched.size() == 2 && busy.size() == 1);
    assert(!ggml_phone_local_host::move_f32(
        mismatched, unmodified, 0));
    assert(mismatched.size() == 2 && unmodified.empty());
    return 0;
}
