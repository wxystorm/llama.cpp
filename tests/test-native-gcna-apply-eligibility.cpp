// Device-free policy regression for native GCNA APPLY_WIDE.
// The legacy G/C/P baseline can be impossible to split into N/A while
// a separate fully-scored candidate is legal. Never discard that candidate.
#include "llama-region-ring-planner.h"
#include <cmath>
#include <cstdio>
#include <limits>

#define CHECK(value) do { if (!(value)) { \
    std::fprintf(stderr, "GCNA_APPLY_ELIGIBILITY_FAIL line=%d condition=%s\n", \
                 __LINE__, #value); \
    return 1; \
} } while (0)

int main() {
    using namespace llama_ring_planner;
    result planned;
    planned.valid = true;
    planned.selected.wall_ms = 120.5;
    planned.top.push_back(planned.selected);

    // Real regression: native search valid=1, legacy baseline_ms=-1.
    CHECK(runtime_plan_eligible_for_apply(planned, true));
    CHECK(!runtime_plan_eligible_for_apply(planned, false));

    // Legacy mode must keep its original valid-baseline requirement.
    planned.baseline_valid = true;
    planned.baseline.wall_ms = 200.0;
    CHECK(runtime_plan_eligible_for_apply(planned, false));
    CHECK(runtime_plan_eligible_for_apply(planned, true));

    // Native mode is not a loophole for invalid or unmeasured candidates.
    planned.valid = false;
    CHECK(!runtime_plan_eligible_for_apply(planned, true));
    planned.valid = true;
    planned.selected.wall_ms = std::numeric_limits<double>::infinity();
    CHECK(!runtime_plan_eligible_for_apply(planned, true));
    planned.selected.wall_ms = 0.0;
    CHECK(!runtime_plan_eligible_for_apply(planned, true));
    planned.selected.wall_ms = -1.0;
    CHECK(!runtime_plan_eligible_for_apply(planned, true));
    planned.selected.wall_ms = 120.5;
    planned.top.clear();
    CHECK(!runtime_plan_eligible_for_apply(planned, true));

    // A legacy baseline with invalid timing cannot be used for speedup.
    planned.top.push_back(planned.selected);
    planned.baseline.wall_ms = -1.0;
    CHECK(!runtime_plan_eligible_for_apply(planned, false));
    CHECK(runtime_plan_eligible_for_apply(planned, true));

    std::puts("GCNA_APPLY_ELIGIBILITY_PASS");
    return 0;
}
