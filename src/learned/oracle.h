#ifndef ORACLE_H
#define ORACLE_H

#include <vector>
#include <array>
#include <cmath>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <cstdint>

class LearnedOracle {
private:
    struct Layer {
        std::vector<std::vector<float>> weights;
        std::vector<float> biases;
    };
    std::vector<Layer> layers;
    bool loaded = false;

    std::vector<float> act_in;
    std::vector<float> act_out;

    std::vector<uint16_t> cdf_cache; // 16-bit quantization halves memory footprint
    bool use_cache = false;
    int min_key_cached = 0;
    int max_key_cached = 0;

    int cache_step = 1;
    float inv_cache_step = 1.0f;
    int cache_shift_amount = 0;

    const size_t MAX_CACHE_ENTRIES = 5000000;
    static constexpr float QUANT_SCALE = 65535.0f;
    static constexpr float INV_QUANT_SCALE = 1.0f / 65535.0f;

    // 2-Way Set-Associative L1 Cache
    struct L1CacheSet {
        int key[2] = {-1, -1};
        float cdf[2] = {0.0f, 0.0f};
        int lru = 0; // Tracks which slot to overwrite
    };

    static constexpr size_t L1_BITS = 8;
    static constexpr size_t L1_SIZE = 1 << L1_BITS;
    // alignas(64) ensures sets map perfectly onto CPU cache lines
    alignas(64) std::array<L1CacheSet, L1_SIZE> l1_cache{};

    // Knuth's Multiplicative Hash
    inline size_t hash_key(int key) const {
        return ((uint32_t)key * 2654435761u) >> (32 - L1_BITS);
    }

public:
    LearnedOracle() {
        act_in.reserve(128);
        act_out.reserve(128);
    }

    void load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "[Oracle] Warning: Could not open " << filename << std::endl;
            return;
        }

        std::ios_base::sync_with_stdio(false);
        std::cin.tie(NULL);

        std::string line;
        Layer current_layer;
        bool processing_weight = false;

        while (std::getline(file, line)) {

            if (line.substr(0, 7) == "MAX_KEY") {
                std::stringstream ss(line.substr(8));
                ss >> max_key_universe;
                if (max_key_universe <= 0.0f) max_key_universe = 1.0f; // Safety fallback
                continue;
            }

            if (line.substr(0, 6) == "WEIGHT") {
                if (!current_layer.weights.empty() || !current_layer.biases.empty()) {
                    layers.push_back(std::move(current_layer));
                    current_layer = Layer();
                }
                processing_weight = true;
                continue;
            } else if (line.substr(0, 4) == "BIAS") {
                processing_weight = false;
                continue;
            }

            std::stringstream ss(line);
            float val;
            std::vector<float> values;
            values.reserve(32);
            while (ss >> val) values.push_back(val);

            if (processing_weight) {
                current_layer.weights.push_back(std::move(values));
            } else {
                current_layer.biases = std::move(values);
            }
        }
        if (!current_layer.weights.empty()) layers.push_back(std::move(current_layer));

        loaded = true;
    }

    void build_cache(int min_k, int max_k) {
        if (!loaded) return;
        min_key_cached = min_k;
        max_key_cached = max_k;

        std::cout << "[Oracle] Building Quantized Lookup Table for range [" << min_k << ", " << max_k << "]..." << std::endl;
        unsigned long long range = (unsigned long long)max_k - min_k + 1;

        cache_step = 1;
        cache_shift_amount = 0;

        if (range > MAX_CACHE_ENTRIES) {
            int required_step = (int)(range / MAX_CACHE_ENTRIES) + 1;
            while (cache_step < required_step) {
                cache_step <<= 1;
                cache_shift_amount++;
            }
        }

        inv_cache_step = 1.0f / (float)cache_step;

        size_t required_size = (size_t)(range >> cache_shift_amount) + 2;
        cdf_cache.resize(required_size);

        size_t idx = 0;
        for (unsigned long long k = 0; k <= range; k += cache_step) {
            if (idx >= cdf_cache.size()) break;
            // Quantize the float (0.0 to 1.0) into a uint16_t (0 to 65535)
            float raw_cdf = run_inference((int)(min_k + k));
            cdf_cache[idx++] = (uint16_t)(raw_cdf * QUANT_SCALE);
        }

        if (idx < cdf_cache.size()) cdf_cache[idx] = 65535;

        // Clear the L1 Cache upon rebuild
        l1_cache.fill(L1CacheSet{});

        std::cout << "[Oracle] Cache built. Size: " << cdf_cache.size() << " entries. Memory reduced via 16-bit quantization." << std::endl;
        use_cache = true;
    }

    float max_key_universe = 1.0f;

    void set_universe_max(float max_val) {
        max_key_universe = max_val > 0 ? max_val : 1.0f;
    }

    float run_inference(int key) {
        if (!loaded) return 0.5f;

        act_in.clear();

        float normalized_key = (float)key / max_key_universe;
        normalized_key = std::max(0.0f, std::min(1.0f, normalized_key));
        act_in.push_back(normalized_key);

        for (size_t i = 0; i < layers.size(); ++i) {
            act_out.clear();
            const auto& w = layers[i].weights;
            const auto& b = layers[i].biases;

            for (size_t j = 0; j < w.size(); ++j) {
                float z = (j < b.size()) ? b[j] : 0.0f;

                // Limit is now safely 1 for the first layer
                const size_t limit = std::min(act_in.size(), w[j].size());

                for (size_t k = 0; k < limit; ++k) {
                    z += act_in[k] * w[j][k];
                }

                if (i == layers.size() - 1) {
                    act_out.push_back(1.0f / (1.0f + std::exp(-z))); // Sigmoid
                } else {
                    act_out.push_back(z > 0.0f ? z : 0.0f); // ReLU
                }
            }
            std::swap(act_in, act_out);
        }

        return act_in.empty() ? 0.5f : act_in[0];
    }

    inline float predict_cdf(int key) {
        // Check 2-Way Set-Associative L1 Cache
        size_t l1_idx = hash_key(key);
        L1CacheSet& set = l1_cache[l1_idx];

        if (set.key[0] == key) return set.cdf[0];
        if (set.key[1] == key) return set.cdf[1];

        float final_cdf = 0.0f;

        if (!use_cache) {
            final_cdf = run_inference(key);
        } else if (key <= min_key_cached) {
            final_cdf = 0.0f;
        } else if (key >= max_key_cached) {
            final_cdf = 1.0f;
        } else {
            unsigned long long offset = (unsigned long long)key - min_key_cached;
            size_t idx = (size_t)(offset >> cache_shift_amount);

            if (idx + 1 >= cdf_cache.size()) {
                final_cdf = 1.0f;
            } else {
                int remainder = (int)(offset & (cache_step - 1));

                // De-quantize back to float
                float y0 = (float)cdf_cache[idx] * INV_QUANT_SCALE;

                if (cache_shift_amount == 0 || remainder == 0) {
                    final_cdf = y0;
                } else {
                    float y1 = (float)cdf_cache[idx + 1] * INV_QUANT_SCALE;
                    final_cdf = y0 + (y1 - y0) * ((float)remainder * inv_cache_step);
                }
            }
        }

        // Populate L1 Cache (Overwrite LRU slot)
        int replace_idx = set.lru;
        set.key[replace_idx] = key;
        set.cdf[replace_idx] = final_cdf;
        set.lru = 1 - replace_idx;

        return final_cdf;
    }

    inline float get_gap_weight(int L, int R, float w_dynamic) {
        float cdf_L = predict_cdf(L);
        float cdf_R = predict_cdf(R);
        float diff = cdf_R - cdf_L;

        return (diff > 1e-6f) ? diff : w_dynamic;
    }
};

#endif

