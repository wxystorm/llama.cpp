// Offline, read-only Region Ring Planner. Reuses exactly the production
// DAG simulator; it NEVER loads a model, starts RPC or runs inference.
// Profiles are real Stage CSV data, supplemented by EXPLICIT transfer and
// memory calibration values. Missing shapes fail closed.
#include "llama-region-ring-real-profile.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace llama_ring_planner;

struct offline_config {
    std::string profile;
    std::string model_id;
    std::string out;
    std::string baseline;
    int layers = 0, tokens = 0, hidden = 0, kv_start = 0;
    int max_npu_region_layers = 1; // current local-only HTP supports one Layer
    double gpu_budget_mib = 0.0, npu_budget_mib = 0.0;
    double gpu_weight_layer_mib = 0.0, npu_weight_layer_mib = 0.0;
    double gpu_kv_layer_mib = 0.0, npu_kv_layer_mib = 0.0;
    double gpu_scratch_mib = 0.0, npu_scratch_mib = 0.0;
    double gpu_h2d_mibps = 0.0, c2p_mibps = 0.0;
    double p2c_mibps = 0.0, local_mibps = 0.0;
    double transfer_latency_ms = 0.0;
    double phone_head_ms = 0.0;
    double e_margin = 0.30;
    int max_dag = 4096, min_dag = 256;
    dag_search_limits limits;
};

static double read_double(const std::map<std::string,std::string> & args,
                          const std::string & key, double def) {
    const auto it = args.find(key);
    if (it == args.end()) { return def; }
    size_t used = 0;
    const double n = std::stod(it->second, &used);
    if (used != it->second.size() || !std::isfinite(n)) {
        throw std::runtime_error("invalid numeric value for " + key);
    }
    return n;
}

static int read_int(const std::map<std::string,std::string> & args,
                    const std::string & key, int def) {
    const double x = read_double(args, key, def);
    if (x < (double) std::numeric_limits<int>::min() ||
            x > (double) std::numeric_limits<int>::max() ||
            x != std::floor(x)) {
        throw std::runtime_error("invalid integer value for " + key);
    }
    return (int) x;
}

static size_t read_size(const std::map<std::string,std::string> & args,
                        const std::string & key, size_t def) {
    const double x = read_double(args, key, (double) def);
    if (x < 1 || x > 1000000 || x != std::floor(x)) {
        throw std::runtime_error("invalid size for " + key);
    }
    return (size_t) x;
}

static offline_config parse_args(int argc, char ** argv) {
    std::map<std::string,std::string> args;
    for (int i = 1; i < argc; i += 2) {
        if (std::string(argv[i]) == "--help") {
            std::cout <<
                "Offline exact DAG Planner. Required:\n"
                "  --profile real-stage-profile.csv --out plan-dir\n"
                "  --model-id Qwen3-14B-Q4_K_M.gguf\n"
                "  --layers 40 --tokens 512 --hidden 5120\n"
                "  --baseline G9,P1,G9,P1,G9,P1,G9,P1\n"
                "  --gpu-budget-mib N --npu-budget-mib N\n"
                "  --gpu-weight-layer-mib N --npu-weight-layer-mib N\n"
                "  --gpu-kv-layer-mib N --npu-kv-layer-mib N\n"
                "  --gpu-h2d-mibps N --c2p-mibps N --p2c-mibps N\n"
                "  --local-mibps N\n"
                "Optional: --e-margin 0.30 --max-dag 4096 --min-dag 256\n"
                "  --first-pool 1500 --first-seeds 500 --beam 64\n"
                "  --final-beam 96 --cpu-dag 256 --local-dag 128\n"
                "  --local-rounds 4 --top 100 --max-npu-region-layers 1\n"
                "  --gpu-scratch-mib 0 --npu-scratch-mib 0\n"
                "  --latency-ms 0.0 --phone-head-ms 0.0 --kv-start 0\n";
            std::exit(0);
        }
        if (i + 1 >= argc || std::string(argv[i]).rfind("--", 0) != 0) {
            throw std::runtime_error("expected --key value pairs; see --help");
        }
        if (!args.emplace(std::string(argv[i]), argv[i + 1]).second) {
            throw std::runtime_error("duplicate argument: " +
                std::string(argv[i]));
        }
    }
    const auto required = [&](const char * name) -> std::string {
        const auto it = args.find(name);
        if (it == args.end() || it->second.empty()) {
            throw std::runtime_error(std::string("missing ") + name);
        }
        return it->second;
    };
    offline_config o;
    o.profile = required("--profile");
    o.out = required("--out");
    o.model_id = required("--model-id");
    o.baseline = required("--baseline");
    o.layers = read_int(args, "--layers", 0);
    o.tokens = read_int(args, "--tokens", 0);
    o.hidden = read_int(args, "--hidden", 0);
    o.kv_start = read_int(args, "--kv-start", 0);
    o.max_npu_region_layers = read_int(args, "--max-npu-region-layers", 1);
    o.gpu_budget_mib = read_double(args, "--gpu-budget-mib", 0);
    o.npu_budget_mib = read_double(args, "--npu-budget-mib", 0);
    o.gpu_weight_layer_mib = read_double(args, "--gpu-weight-layer-mib", 0);
    o.npu_weight_layer_mib = read_double(args, "--npu-weight-layer-mib", 0);
    o.gpu_kv_layer_mib = read_double(args, "--gpu-kv-layer-mib", 0);
    o.npu_kv_layer_mib = read_double(args, "--npu-kv-layer-mib", 0);
    o.gpu_scratch_mib = read_double(args, "--gpu-scratch-mib", 0);
    o.npu_scratch_mib = read_double(args, "--npu-scratch-mib", 0);
    o.gpu_h2d_mibps = read_double(args, "--gpu-h2d-mibps", 0);
    o.c2p_mibps = read_double(args, "--c2p-mibps", 0);
    o.p2c_mibps = read_double(args, "--p2c-mibps", 0);
    o.local_mibps = read_double(args, "--local-mibps", 0);
    o.transfer_latency_ms = read_double(args, "--latency-ms", 0);
    o.phone_head_ms = read_double(args, "--phone-head-ms", 0);
    o.e_margin = read_double(args, "--e-margin", 0.30);
    o.max_dag = read_int(args, "--max-dag", 4096);
    o.min_dag = read_int(args, "--min-dag", 256);
    o.limits.first_cycle_e_pool = read_size(args, "--first-pool", 1500);
    o.limits.first_cycle_seeds = read_size(args, "--first-seeds", 500);
    o.limits.seed_geometry_diversity_per_n = read_size(args, "--geometry", 28);
    o.limits.seed_chunk_diversity = read_size(args, "--chunk-diversity", 96);
    o.limits.prefix_beam_per_seed = read_size(args, "--beam", 64);
    o.limits.final_beam_per_seed = read_size(args, "--final-beam", 96);
    o.limits.cpu_optional_dag_cap = read_size(args, "--cpu-dag", 256);
    o.limits.cpu_optional_anchors_per_cycles =
        read_int(args, "--cpu-anchors", 12);
    o.limits.local_rounds = read_int(args, "--local-rounds", 4);
    o.limits.local_simulations_per_round = read_size(args, "--local-dag", 128);
    o.limits.local_max_anchors = read_size(args, "--local-anchors", 32);
    o.limits.throughput_per_cycle = read_size(args, "--throughput-dag", 12);
    o.limits.retain_top = read_size(args, "--top", 100);
    if (o.layers < 6 || o.tokens < 1 || o.hidden < 1 || o.kv_start < 0 ||
            o.max_npu_region_layers < 1 || o.e_margin < 0 ||
            o.max_dag <= 0 || o.min_dag < 0 ||
            !dag_limits_valid(o.limits) ||
            o.gpu_budget_mib <= 0 || o.npu_budget_mib <= 0 ||
            o.gpu_weight_layer_mib <= 0 || o.npu_weight_layer_mib <= 0 ||
            o.gpu_kv_layer_mib < 0 || o.npu_kv_layer_mib < 0 ||
            o.gpu_scratch_mib < 0 || o.npu_scratch_mib < 0 ||
            o.gpu_h2d_mibps <= 0 || o.c2p_mibps <= 0 ||
            o.p2c_mibps <= 0 || o.local_mibps <= 0 ||
            o.transfer_latency_ms < 0 || o.phone_head_ms < 0) {
        throw std::runtime_error(
            "missing/invalid calibration or search budget (see --help)");
    }
    return o;
}

static bool memory_fits(
        const std::vector<llama_region_ring_region> & regions,
        const offline_config & o) {
    std::array<double, 2> gpu_bank {}, npu_bank {};
    int gpu_layers = 0, npu_layers = 0;
    size_t g = 0, p = 0;
    for (const auto & r : regions) {
        const int width = r.layer_end - r.layer_begin;
        if (width <= 0) { return false; }
        if (r.device == device::GPU) {
            gpu_bank[g++ % 2] = std::max(
                gpu_bank[(g - 1) % 2],
                width * o.gpu_weight_layer_mib);
            gpu_layers += width;
        } else if (r.device == device::PHONE) {
            if (width > o.max_npu_region_layers) { return false; }
            npu_bank[p++ % 2] = std::max(
                npu_bank[(p - 1) % 2],
                width * o.npu_weight_layer_mib);
            npu_layers += width;
        }
    }
    return gpu_bank[0] + gpu_bank[1] +
               gpu_layers * o.gpu_kv_layer_mib +
               o.gpu_scratch_mib <= o.gpu_budget_mib &&
           npu_bank[0] + npu_bank[1] +
               npu_layers * o.npu_kv_layer_mib +
               o.npu_scratch_mib <= o.npu_budget_mib;
}

static std::string json_quote(const std::string & s) {
    std::string out = "\"";
    static const char * hex = "0123456789abcdef";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') {
            out.push_back('\\');
            out.push_back((char) c);
        } else if (c < 0x20) {
            out += "\\u00";
            out.push_back(hex[c >> 4]);
            out.push_back(hex[c & 15]);
        } else {
            out.push_back((char) c);
        }
    }
    out += '"';
    return out;
}

static std::string npu_regions_csv(
        const std::vector<llama_region_ring_region> & regions) {
    std::string s;
    for (size_t i = 0; i < regions.size(); ++i) {
        if (regions[i].device != device::PHONE) { continue; }
        if (!s.empty()) { s += ','; }
        s += std::to_string(i);
    }
    return s;
}

int main(int argc, char ** argv) {
    try {
        const offline_config o = parse_args(argc, argv);
        real_stage_profile measurements;
        std::string error;
        if (!measurements.load(o.profile, error)) {
            throw std::runtime_error("cannot load real Stage Profile: " + error);
        }
        auto baseline = llama_region_ring_parse_apply_wide_layout(
            o.baseline, o.layers);
        if (baseline.empty() ||
                baseline.back().device != device::PHONE ||
                !memory_fits(baseline, o)) {
            throw std::runtime_error(
                "baseline layout is not HTP-supported or exceeds two-bank budget");
        }

        model m;
        m.layers = o.layers;
        m.tokens = o.tokens;
        m.kv_start = o.kv_start;
        m.hidden = o.hidden;
        m.gpu_weight_streaming = true;
        m.pc_compute_gate = false;
        m.shared_phone_link = false;
        m.phone_full_return_stream = true;
        m.phone_autonomous_prebuild = true;
        m.phone_autonomous_prepare_overlap = true;
        m.phone_autonomous_final = true;
        m.phone_auto_final_head_ms = o.phone_head_ms;
        m.compute_ms = [&](device d, int layers, int x, int kv, double & ms) {
            return measurements.estimate(
                d, layers, x, kv, false, real_stage_phase::COMPUTE, ms) ||
                (d == device::PHONE && measurements.estimate(
                    d, layers, x, kv, true, real_stage_phase::COMPUTE, ms));
        };
        m.compute_region_ms = [&](device d, int begin, int end,
                                  int x, int kv, bool terminal, double & ms) {
            return measurements.estimate(
                d, end - begin, x, kv, terminal,
                real_stage_phase::COMPUTE, ms);
        };
        m.prepare_ms = [&](device d, int layers, int x, int kv, double & ms) {
            return measurements.estimate(
                d, layers, x, kv, false, real_stage_phase::PREPARE, ms) ||
                (d == device::PHONE && measurements.estimate(
                    d, layers, x, kv, true, real_stage_phase::PREPARE, ms));
        };
        m.prepare_region_ms = [&](device d, int begin, int end,
                                  int x, int kv, bool terminal, double & ms) {
            return measurements.estimate(
                d, end - begin, x, kv, terminal,
                real_stage_phase::PREPARE, ms);
        };
        m.transfer_ms = [&](device a, device b, size_t bytes, double & ms) {
            if (a == b) { ms = 0.0; return true; }
            const double speed =
                a == device::PHONE ? o.p2c_mibps :
                b == device::PHONE ? o.c2p_mibps : o.local_mibps;
            ms = o.transfer_latency_ms +
                (double) bytes / (1024.0 * 1024.0) /
                speed * 1000.0;
            return std::isfinite(ms);
        };
        m.gpu_region_weight_bytes =
            [&](int begin, int end, size_t & n) {
                if (end <= begin) { return false; }
                const long double bytes =
                    (long double) (end - begin) * o.gpu_weight_layer_mib *
                    1024.0L * 1024.0L;
                if (bytes > (long double)
                        std::numeric_limits<size_t>::max()) { return false; }
                n = (size_t) bytes;
                return true;
            };
        m.gpu_weight_h2d_ms = [&](size_t bytes, double & ms) {
            ms = (double) bytes / (1024.0 * 1024.0) /
                o.gpu_h2d_mibps * 1000.0;
            return true;
        };
        m.region_memory_feasible = [&](const auto & regions) {
            return memory_fits(regions, o);
        };
        m.memory_feasible = [&](const candidate & c) {
            std::vector<llama_region_ring_region> regions;
            return make_regions(c, o.layers, regions) &&
                memory_fits(regions, o);
        };
        const result result = search_runtime_applicable(
            m, baseline, 128, o.max_dag, o.e_margin, o.min_dag, o.limits);
        if (!result.valid || result.top.empty()) {
            throw std::runtime_error(
                "no feasible DAG: collect more Stage depth/Token samples, "
                "or check NPU/VRAM limits (no synthetic extrapolation)");
        }
        const auto & chosen = result.selected;
        std::vector<llama_region_ring_region> chosen_regions;
        if (!make_regions(chosen, o.layers, chosen_regions) ||
                !memory_fits(chosen_regions, o)) {
            throw std::runtime_error("selected DAG failed memory recheck");
        }
        fs::create_directories(o.out);
        const fs::path out(o.out);
        {
            std::ofstream json(out / "plan.json");
            if (!json) { throw std::runtime_error("cannot write plan.json"); }
            json << std::fixed << std::setprecision(3);
            json << "{\n"
                 << "  \"schema\": 1,\n"
                 << "  \"model_id\": " << json_quote(o.model_id) << ",\n"
                 << "  \"source_profile\": " << json_quote(o.profile) << ",\n"
                 << "  \"planner\": \"offline-exact-dag\",\n"
                 << "  \"prediction_scope\": \"COMPUTE_AND_NETWORK_ONLY_NPU_REFILL_NOT_SIMULATED\",\n"
                 << "  \"layers\": " << o.layers << ",\n"
                 << "  \"tokens\": " << o.tokens << ",\n"
                 << "  \"hidden\": " << o.hidden << ",\n"
                 << "  \"layout\": "
                 << json_quote(llama_region_ring_layout_string(chosen_regions))
                 << ",\n"
                 << "  \"xc\": " << chosen.xc << ",\n"
                 << "  \"xg\": " << chosen.xg << ",\n"
                 << "  \"xp\": " << chosen.xp << ",\n"
                 << "  \"phone_npu_regions\": "
                 << json_quote(npu_regions_csv(chosen_regions)) << ",\n"
                 << "  \"max_npu_region_layers\": "
                 << o.max_npu_region_layers << ",\n"
                 << "  \"gpu_budget_mib\": " << o.gpu_budget_mib << ",\n"
                 << "  \"npu_budget_mib\": " << o.npu_budget_mib << ",\n"
                 << "  \"predicted_ms_lower_bound\": " << chosen.wall_ms << ",\n"
                 << "  \"dag_simulated\": " << result.simulated << ",\n"
                 << "  \"dag_profiled\": " << result.profiled << ",\n"
                 << "  \"dag_memory_rejected\": " << result.rejected_memory << "\n"
                 << "}\n";
        }
        {
            std::ofstream csv(out / "candidates.csv");
            if (!csv) {
                throw std::runtime_error("cannot write candidates.csv");
            }
            csv << "rank,layout,xc,xg,xp,predicted_lower_bound_ms,"
                   "balance,phone_npu_regions\n";
            csv << std::fixed << std::setprecision(3);
            for (size_t i = 0; i < result.top.size(); ++i) {
                std::vector<llama_region_ring_region> rs;
                if (!make_regions(result.top[i], o.layers, rs)) {
                    throw std::runtime_error("candidate cannot render");
                }
                const auto & c = result.top[i];
                csv << i + 1 << ",\""
                    << llama_region_ring_layout_string(rs) << "\","
                    << c.xc << ',' << c.xg << ',' << c.xp << ','
                    << c.wall_ms << ',' << c.ring_error_max
                    << ",\"" << npu_regions_csv(rs) << "\"\n";
            }
        }
        std::cout << "[OFFLINE_PLAN] model=" << o.model_id
                  << " profiled=" << result.profiled
                  << " DAG=" << result.simulated
                  << " layout=" << llama_region_ring_layout_string(chosen_regions)
                  << " XC=" << chosen.xc << " XG=" << chosen.xg
                  << " XP=" << chosen.xp
                  << " predicted_lower_bound_ms=" << chosen.wall_ms
                  << " status=PLAN_ONLY\n";
        std::cout <<
            "[OFFLINE_PLAN] NOTE: predicted_ms excludes NPU weight-refill "
            "critical path until the real two-Bank executor is measured. "
            "Do not use it as a validated prefill forecast.\n";
        std::cout << "[OFFLINE_PLAN] file="
                  << (out / "plan.json").string() << '\n';
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "[OFFLINE_PLAN] status=FAIL reason=" << e.what() << '\n';
        return 1;
    }
}
