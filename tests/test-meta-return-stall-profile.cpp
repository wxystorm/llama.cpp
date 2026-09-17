#include "ggml-backend.h"

#include <cassert>

int main() {
    ggml_backend_meta_tensor_profile profile {};

    profile.lane_reuse_wait_count = 3;
    profile.lane_reuse_wait_us = 7000;
    profile.lane_reuse_wait_max_us = 4000;
    profile.lane_reuse_wait_count_by_lane[0] = 2;
    profile.lane_reuse_wait_count_by_lane[1] = 1;
    profile.lane_reuse_wait_us_by_lane[0] = 5000;
    profile.lane_reuse_wait_us_by_lane[1] = 2000;
    profile.layer_barrier_wait_count = 1;
    profile.layer_barrier_wait_us = 6000;
    profile.layer_barrier_wait_max_us = 6000;
    profile.layer_barrier_wait_max_layer = 23;
    profile.layer_barrier_wait_max_pending = 1;
    profile.layer_barrier_wait_max_last_lane = 0;

    assert(profile.lane_reuse_wait_count == 3);
    assert(profile.lane_reuse_wait_us_by_lane[0] + profile.lane_reuse_wait_us_by_lane[1] ==
           profile.lane_reuse_wait_us);
    assert(profile.layer_barrier_wait_max_us == profile.layer_barrier_wait_us);
    assert(profile.layer_barrier_wait_max_layer == 23);
    assert(profile.layer_barrier_wait_max_pending == 1);
    assert(profile.layer_barrier_wait_max_last_lane == 0);

    return 0;
}
