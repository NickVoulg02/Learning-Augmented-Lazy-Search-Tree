#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lazy-search-tree.cpp"

#include "../../shared/benchmark_runner.hpp"

template <>
lazy_search_tree<int, CountingComp<int>> construct_tree<lazy_search_tree<int, CountingComp<int>>>(int max_key_hint) {
    return lazy_search_tree<int, CountingComp<int>>(max_key_hint);
}

namespace py = pybind11;

// Define the global counter
thread_local size_t global_cmp_count = 0;

PYBIND11_MODULE(lazy_tree_learned, m) {
    m.doc() = "Learned Lazy Search Tree Bindings";

    py::class_<BenchmarkResult>(m, "BenchmarkResult", py::module_local())
        .def_readonly("insert_time", &BenchmarkResult::insert_time)
        .def_readonly("query_time", &BenchmarkResult::query_time)
        .def_readonly("total_time", &BenchmarkResult::total_time)
        .def_readonly("tree_size", &BenchmarkResult::tree_size)
        .def_readonly("checksum", &BenchmarkResult::checksum)
        .def_readonly("total_memory_kb", &BenchmarkResult::total_memory_kb)
        .def_readonly("oracle_memory_kb", &BenchmarkResult::oracle_memory_kb)
        .def_readonly("tree_memory_kb", &BenchmarkResult::tree_memory_kb)
        .def_readonly("insert_comparisons", &BenchmarkResult::insert_comparisons) 
        .def_readonly("query_comparisons", &BenchmarkResult::query_comparisons)
        .def_readonly("p99_query_time", &BenchmarkResult::p99_query_time);

    m.def("run_custom", [](const std::string& tree_type, const std::vector<int>& preload, const std::vector<int>& inserts, const std::vector<int>& queries) {
        int max_key_hint = 0;
        if (!preload.empty()) { max_key_hint = *std::max_element(preload.begin(), preload.end()); }

        if (tree_type == "BTree") {
            return run_custom_benchmark_cpp<FastBTree<int, CountingComp<int>>>(preload, inserts, queries, max_key_hint);
        } else if (tree_type == "Set") {
            return run_custom_benchmark_cpp<std::set<int, CountingComp<int>>>(preload, inserts, queries, max_key_hint);
        } else if (tree_type == "Treap") {
            return run_custom_benchmark_cpp<treap<int, CountingComp<int>>>(preload, inserts, queries, max_key_hint);
        } else {
            return run_custom_benchmark_cpp<lazy_search_tree<int, CountingComp<int>>>(preload, inserts, queries, max_key_hint);
        }
    }, "Run batched custom benchmark");

    m.def("run_custom_interleaved", [](const std::string& tree_type, const std::vector<int>& preload, const std::vector<int>& inserts, const std::vector<int>& queries) {
        int max_key_hint = 0;
        if (!preload.empty()) {
            max_key_hint = *std::max_element(preload.begin(), preload.end());
        }

        if (tree_type == "BTree") {
            return run_custom_interleaved_cpp<FastBTree<int, CountingComp<int>>>(preload, inserts, queries, max_key_hint);
        } else if (tree_type == "Set") {
            return run_custom_interleaved_cpp<std::set<int, CountingComp<int>>>(preload, inserts, queries, max_key_hint);
        } else if (tree_type == "Treap") {
            return run_custom_interleaved_cpp<treap<int, CountingComp<int>>>(preload, inserts, queries, max_key_hint);
        } else {
            return run_custom_interleaved_cpp<lazy_search_tree<int, CountingComp<int>>>(preload, inserts, queries, max_key_hint);
        }
    }, "Run interleaved custom benchmark");

    m.def("run_custom_interleaved_delete", [](const std::string& tree_type, const std::vector<int>& preload, const std::vector<int>& inserts, const std::vector<int>& queries) {
        int max_key_hint = 0;
        if (!preload.empty()) { max_key_hint = *std::max_element(preload.begin(), preload.end()); }

        if (tree_type == "BTree") {
            return run_custom_interleaved_delete_cpp<FastBTree<int, CountingComp<int>>>(preload, inserts, queries, max_key_hint);
        } else if (tree_type == "Set") {
            return run_custom_interleaved_delete_cpp<std::set<int, CountingComp<int>>>(preload, inserts, queries, max_key_hint);
        } else if (tree_type == "Treap") {
            return run_custom_interleaved_delete_cpp<treap<int, CountingComp<int>>>(preload, inserts, queries, max_key_hint);
        } else {
            return run_custom_interleaved_delete_cpp<lazy_search_tree<int, CountingComp<int>>>(preload, inserts, queries, max_key_hint);
        }
    }, "Run interleaved benchmark with deletes");
}
