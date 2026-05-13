#ifndef BENCHMARK_RUNNER_HPP
#define BENCHMARK_RUNNER_HPP

#include <chrono>
#include <random>
#include <set>
#include <string>
#include <numeric>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <malloc.h>

#include <tlx/container/btree_multiset.hpp>

namespace tlx {
    // Implement the missing tlx assertion handler to satisfy the linker
    // and safely pass B-Tree crashes up to Python as RuntimeErrors.
    void die_with_message(const char* msg, const char* file, size_t line) {
        std::string error_msg = "TLX B-Tree Error: ";
        if (msg) error_msg += msg;
        error_msg += " (";
        if (file) error_msg += file;
        error_msg += ":" + std::to_string(line) + ")";
        throw std::runtime_error(error_msg);
    }
}

extern thread_local size_t global_cmp_count;

template <typename T>
struct CountingComp {
    bool operator()(const T& a, const T& b) const {
        global_cmp_count++;
        return a < b;
    }
};

#ifdef _WIN32
    // Windows Memory Tracking
    #include <windows.h>
    #include <psapi.h>
    #pragma comment(lib, "psapi.lib")

    long get_current_rss_kb() {
        PROCESS_MEMORY_COUNTERS_EX pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
            return (long)(pmc.PagefileUsage / 1024);
        }
        return 0;
    }
#else
    // Linux/POSIX Memory Tracking
    #include <fstream>
    #include <string>
    #include <malloc.h>

    void _heapmin() {
        malloc_trim(0);
    }

    long get_current_rss_kb() {
        std::ifstream status_file("/proc/self/status");
        std::string line;
        while (std::getline(status_file, line)) {
            if (line.compare(0, 6, "VmRSS:") == 0) {
                long rss = 0;
                sscanf(line.c_str(), "VmRSS: %ld kB", &rss);
                return rss;
            }
        }
        return 0;
    }
#endif

template <typename T>
T construct_tree(int max_key_hint) {
    (void)max_key_hint;
    return T();
}

template <typename T, typename Comp = std::less<T>>
struct FastBTree {
    tlx::btree_multiset<T, Comp> tree;
    void insert(const T& key) { tree.insert(key); }
    void erase(const T& key) { tree.erase(key); }
    size_t count(const T& key) { return (tree.find(key) != tree.end()) ? 1 : 0; }
    size_t size() const { return tree.size(); }
};

struct BenchmarkResult {
    double insert_time;
    double query_time;
    double total_time;
    size_t tree_size;
    size_t checksum;
    long total_memory_kb;
    long oracle_memory_kb;
    long tree_memory_kb;
    size_t insert_comparisons;
    size_t query_comparisons;
    double p99_query_time;
};

template <typename Func>
double measure_time(Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}


template <typename TreeType>
BenchmarkResult run_custom_benchmark_cpp(
    const std::vector<int>& preload,
    const std::vector<int>& inserts,
    const std::vector<int>& queries,
    int max_key_hint) {

    _heapmin();

    long start_mem = get_current_rss_kb();
    TreeType tree = construct_tree<TreeType>(max_key_hint);
    long post_init_mem = get_current_rss_kb();

    BenchmarkResult res;
    res.query_time = 0.0;
    res.p99_query_time = 0.0;

    for (int x : preload) {
        tree.insert(x);
    }

    global_cmp_count = 0;
    res.insert_time = measure_time([&]() {
        for (int x : inserts) tree.insert(x);
    });
    res.insert_comparisons = global_cmp_count;

    size_t total_hits = 0;
    global_cmp_count = 0; // Reset before queries

    if (!queries.empty()) {
        auto q_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < queries.size(); ++i) {
            if (tree.count(queries[i])) total_hits++;
        }
        auto q_end = std::chrono::high_resolution_clock::now();
        res.query_time = std::chrono::duration<double>(q_end - q_start).count();
    }

    res.query_comparisons = global_cmp_count;
    res.checksum = total_hits;
    res.total_time = res.insert_time + res.query_time;
    res.tree_size = tree.size();

    long final_mem = get_current_rss_kb();
    res.oracle_memory_kb = post_init_mem - start_mem;
    res.total_memory_kb = final_mem - start_mem;
    res.tree_memory_kb = final_mem - post_init_mem;

    return res;
}

template <typename TreeType>
BenchmarkResult run_custom_interleaved_cpp(
    const std::vector<int>& preload,
    const std::vector<int>& inserts,
    const std::vector<int>& queries,
    int max_key_hint) {

    _heapmin();

    long start_mem = get_current_rss_kb();
    TreeType tree = construct_tree<TreeType>(max_key_hint);
    long post_init_mem = get_current_rss_kb();
    BenchmarkResult res;

    for (const int& x : preload) {
        tree.insert(x);
    }

    size_t insert_n = inserts.size();
    size_t query_n = queries.size();

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> d_decision(0, (int)insert_n - 1);

    size_t hits = 0;
    size_t q_idx = 0;

    std::vector<double> query_latencies;
    query_latencies.reserve(query_n);

    global_cmp_count = 0;
    size_t insert_cmps = 0;
    size_t query_cmps = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for(size_t i=0; i<insert_n; ++i) {
        global_cmp_count = 0;
        tree.insert(inserts[i]);
        insert_cmps += global_cmp_count;

        if (q_idx < query_n && d_decision(rng) < query_n) {
            // Track individual query
            global_cmp_count = 0;
            auto q_start = std::chrono::high_resolution_clock::now();
            bool found = tree.count(queries[q_idx]);
            auto q_end = std::chrono::high_resolution_clock::now();

            query_cmps += global_cmp_count;
            if (found) hits++;
            query_latencies.push_back(std::chrono::duration<double>(q_end - q_start).count());
            q_idx++;
        }
    }

    while(q_idx < query_n) {
        global_cmp_count = 0;
        auto q_start = std::chrono::high_resolution_clock::now();
        bool found = tree.count(queries[q_idx]);
        auto q_end = std::chrono::high_resolution_clock::now();

        query_cmps += global_cmp_count;
        if (found) hits++;
        query_latencies.push_back(std::chrono::duration<double>(q_end - q_start).count());
        q_idx++;
    }

    auto end = std::chrono::high_resolution_clock::now();

    if (!query_latencies.empty()) {
        std::sort(query_latencies.begin(), query_latencies.end());
        size_t p99_idx = (size_t)(0.99 * query_latencies.size());
        if (p99_idx >= query_latencies.size()) p99_idx = query_latencies.size() - 1;
        res.p99_query_time = query_latencies[p99_idx];
    } else {
        res.p99_query_time = 0.0;
    }

    res.insert_comparisons = insert_cmps;
    res.query_comparisons = query_cmps;
    res.total_time = std::chrono::duration<double>(end - start).count();
    res.insert_time = res.total_time;
    res.query_time = 0.0;
    res.checksum = hits;
    res.tree_size = tree.size();

    long final_mem = get_current_rss_kb();
    res.oracle_memory_kb = post_init_mem - start_mem;
    res.total_memory_kb = final_mem - start_mem;
    res.tree_memory_kb = final_mem - post_init_mem;

    return res;
}

template <typename TreeType>
BenchmarkResult run_custom_interleaved_delete_cpp(
    const std::vector<int>& preload,
    const std::vector<int>& inserts,
    const std::vector<int>& queries,
    int max_key_hint) {

    _heapmin();

    long start_mem = get_current_rss_kb();
    TreeType tree = construct_tree<TreeType>(max_key_hint);
    long post_init_mem = get_current_rss_kb();
    BenchmarkResult res;

    // Track active keys so we always delete something that exists
    std::vector<int> active_keys = preload;
    active_keys.reserve(preload.size() + inserts.size());
    for (const int& x : preload) {
        tree.insert(x);
    }

    size_t num_queries = queries.size();
    size_t num_inserts = inserts.size();
    size_t q_idx = 0;
    size_t i_idx = 0;

    size_t hits = 0;
    size_t del_count = 0;
    size_t ins_cmps = 0;
    size_t qry_cmps = 0;

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    auto start = std::chrono::high_resolution_clock::now();

    // 60% Query, 20% Insert, 20% Delete
    while (q_idx < num_queries || i_idx < num_inserts) {
        double r = dist(rng);
        if ((r < 0.60 || i_idx >= num_inserts) && q_idx < num_queries) {
            global_cmp_count = 0;
            if (tree.count(queries[q_idx++])) hits++;
            qry_cmps += global_cmp_count;
        }
        else if ((r < 0.80 || q_idx >= num_queries) && i_idx < num_inserts) {
            int k = inserts[i_idx++];
            global_cmp_count = 0;
            tree.insert(k);
            ins_cmps += global_cmp_count;
            active_keys.push_back(k);
        }
        else if (!active_keys.empty()) {
            size_t del_idx = rng() % active_keys.size();
            int k = active_keys[del_idx];
            active_keys[del_idx] = active_keys.back();
            active_keys.pop_back();

            global_cmp_count = 0;
            tree.erase(k);
            qry_cmps += global_cmp_count;
            del_count++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    res.total_time = std::chrono::duration<double>(end - start).count();
    res.insert_time = res.total_time;
    res.query_time = 0.0;
    res.checksum = hits + del_count;
    res.tree_size = tree.size();
    res.insert_comparisons = ins_cmps;
    res.query_comparisons = qry_cmps;

    res.p99_query_time = (double)(q_idx + i_idx + del_count);

    long final_mem = get_current_rss_kb();
    res.oracle_memory_kb = post_init_mem - start_mem;
    res.total_memory_kb = final_mem - start_mem;
    res.tree_memory_kb = final_mem - post_init_mem;

    return res;
}

#endif