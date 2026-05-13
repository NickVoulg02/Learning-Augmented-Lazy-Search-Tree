#ifndef LAZY_SEARCH_TREE
#define LAZY_SEARCH_TREE

#define INF 1000000000
#include "treap.cpp"
#include "oracle.h"
#include <list>
#include <memory>
#include <limits>
#include <tuple>
#include <algorithm>

using namespace std;

template<typename T, typename Comp = std::less<T>>
class lazy_search_tree {
private:
    Comp comp;
    size_t lst_size;

public:
    static LearnedOracle oracle;
    static unsigned long total_elements_global;

private:
    class gap {
    private:
        Comp comp;
        class interval {
        private:
            Comp comp;
            T max_e;
            T min_e;
            size_t int_size;
            

            list<vector<T>> elements;

        public:
            T sample() {
                if (size() == 0) return T();
                int idx = FastRNG::next() % size();

                for (vector<T>& vec : elements) {
                    if (idx < (int)vec.size()) return vec[idx];
                    idx -= static_cast<int>(vec.size());
                }
                return T();
            }

            void merge(shared_ptr<interval> other) {
                if (other->int_size == 0) return;
                int_size += other->int_size;

                if (comp(max_e, other->max_e)) max_e = other->max_e;
                if (comp(other->min_e, min_e)) min_e = other->min_e;

                elements.splice(elements.end(), other->elements);
                other->int_size = 0;
            }

            void insert(const T &element) {
                elements.front().emplace_back(element);
                if (comp(max_e, element)) max_e = element;
                if (comp(element, min_e)) min_e = element;
                ++int_size;
            }

            interval(vector<T> &starting_elements, Comp c) : comp(c) {
                int_size = starting_elements.size();
                if (!starting_elements.empty()) {
                    elements.emplace_back(starting_elements);
                    max_e = starting_elements[0];
                    min_e = starting_elements[0];
                    for (const auto& e : starting_elements) {
                        if (comp(max_e, e)) max_e = e;
                        if (comp(e, min_e)) min_e = e;
                    }
                }
            }

            interval(const T &element, Comp c) : comp(c) {
                int_size = 1;
                elements.emplace_back(vector<T>({element}));
                max_e = element;
                min_e = element;
            }

            tuple<shared_ptr<interval>, shared_ptr<interval>, bool> pivot(const T &p) {
                if (elements.size() > 1) {
                    vector<T> flat;
                    flat.reserve(int_size);
                    for (auto& vec : elements) {
                        flat.insert(flat.end(), std::make_move_iterator(vec.begin()), std::make_move_iterator(vec.end()));
                    }
                    elements.clear();
                    elements.emplace_back(std::move(flat));
                }

                vector<T>& data = elements.front();
                bool found = false;

                auto bound = std::partition(data.begin(), data.end(), [&](const T& e) {
                    if (!comp(e, p) && !comp(p, e)) {
                        found = true;
                        // Distribute duplicates evenly using your RNG logic
                        return (FastRNG::next() & 1) == 0;
                    }
                    return comp(e, p);
                });

                vector<T> lesser_vec(std::make_move_iterator(data.begin()), std::make_move_iterator(bound));
                vector<T> greater_vec(std::make_move_iterator(bound), std::make_move_iterator(data.end()));

                return make_tuple(make_shared<interval>(lesser_vec, comp),
                                  make_shared<interval>(greater_vec, comp),
                                  found);
            }

            bool erase(const T &key) {
                for (auto list_it = elements.begin(); list_it != elements.end(); ++list_it) {
                    auto& vec = *list_it;
                    for (auto vec_it = vec.begin(); vec_it != vec.end(); ++vec_it) {

                        if (!comp(*vec_it, key) && !comp(key, *vec_it)) {
                            std::iter_swap(vec_it, vec.end() - 1);
                            vec.pop_back();
                            --int_size;

                            if (vec.empty()) {
                                elements.erase(list_it);
                            }

                            if (int_size > 0 && (!comp(min_e, key) || !comp(key, max_e))) {
                                auto it = elements.begin();
                                if (it != elements.end() && !it->empty()) {
                                    max_e = min_e = (*it)[0];
                                    for (const auto& v : elements) {
                                        for (const auto& e : v) {
                                            if (comp(max_e, e)) max_e = e;
                                            if (comp(e, min_e)) min_e = e;
                                        }
                                    }
                                }
                            }
                            return true;
                        }
                    }
                }
                return false;
            }

            bool operator< (const interval& other) const {
                if (comp(max_e, other.max_e)) return true;
                if (comp(other.max_e, max_e)) return false;
                return comp(min_e, other.min_e);
            }

            size_t size() const { return int_size; }
            T get_max() { return max_e; }
            T get_min() { return min_e; }
            bool empty( ) const { return size() == 0; }
        };

        size_t gap_size;
        int last_left_idx = 0;
        T gap_max;
        T gap_min;
        vector<shared_ptr<interval>> intervals;

        void refresh_stats() {
            if (intervals.empty()) return;
            gap_max = intervals[0]->get_max();
            gap_min = intervals[0]->get_min();

            for (auto& ints : intervals) {
                T i_max = ints->get_max();
                T i_min = ints->get_min();
                if (comp(gap_max, i_max)) gap_max = i_max;
                if (comp(i_min, gap_min)) gap_min = i_min;
            }
        }

    public:
        size_t last_priority_update_size = 0;
        
        gap(vector<shared_ptr<interval>> &intervals, Comp c) : comp(c) {
            gap_size = 0;
            this->intervals.reserve(intervals.size());
            for (shared_ptr<interval> g_int : intervals) {
                if (!g_int->empty()) {
                    this->intervals.emplace_back(g_int);
                    gap_size += g_int->size();
                }
            }
            rebalance();
            refresh_stats();
        }

        int getIntervalIdx(const T &key) {
            if (intervals.empty()) return 0;
            int lo = last_left_idx, hi, mult;
            bool init = !comp(intervals[last_left_idx]->get_max(), key);
            if (init) mult = -1; else mult = 1;
            int n_intervals = (int)intervals.size();

            for (int i = 0; ; ++i) {
                hi = lo + mult * (1 << i);
                if (hi < 0) { hi = -1; break; }
                if (hi >= n_intervals) { hi = n_intervals; break; }
                bool current_leq = !comp(intervals[hi]->get_max(), key);
                if (init != current_leq) break;
            }
            for (;;) {
                if (abs(hi - lo) <= 1) {
                    if (init || hi == n_intervals) return lo;
                    else return hi;
                }
                int mid = (lo + hi) / 2;
                bool mid_leq = !comp(intervals[mid]->get_max(), key);
                if (init == mid_leq) lo = mid; else hi = mid;
            }
        }

        T pick_pivot(int sample_size, shared_ptr<interval> g_int) {
            vector<T> pivots;
            pivots.reserve(sample_size);
            for (int i = 0; i < sample_size; ++i) pivots.push_back(g_int->sample());
            std::sort(pivots.begin(), pivots.end(), comp);
            return pivots[sample_size / 2];
        }

        vector<shared_ptr<interval>> split(shared_ptr<interval> g_int, bool recurse_left, int n_recursions) {
            if (n_recursions == 0 || g_int->size() <= 1) {
                vector<shared_ptr<interval>> temp;
                temp.reserve(1);
                temp.emplace_back(g_int);
                return temp;
            }

            T p = pick_pivot(5, g_int);
            auto result_split = g_int->pivot(p);
            shared_ptr<interval> lesser = get<0>(result_split);
            shared_ptr<interval> greater = get<1>(result_split);

            vector<shared_ptr<interval>> result;
            if (recurse_left) {
                result = split(lesser, true, n_recursions - 1);
                result.emplace_back(greater);
            } else {
                result.reserve(2);
                result.emplace_back(lesser);
                vector<shared_ptr<interval>> temp = split(greater, false, n_recursions - 1);
                result.insert(result.end(), temp.begin(), temp.end());
            }
            return result;
        }

        gap(const T &key_max, const T &key_min, Comp c) : comp(c), gap_size(0) {
            gap_max = key_max;
            gap_min = key_min;
        }

        gap(const T &key, Comp c) : comp(c) {
            gap_size = 1;
            intervals.emplace_back(make_shared<interval>(key, c));
            gap_max = key;
            gap_min = key;
        }

        bool operator< (const gap& other) const {
            if (comp(gap_max, other.gap_max)) return true;
            if (comp(other.gap_max, gap_max)) return false;
            return comp(gap_min, other.gap_min);
        }

        void insert(const T &key) {
            intervals[getIntervalIdx(key)]->insert(key);
            ++gap_size;
            if (comp(gap_max, key)) gap_max = key;
            if (comp(key, gap_min)) gap_min = key;
        }

        tuple<gap, gap, bool> restructure(const T &key, int n_recursions) {
            int int_idx = getIntervalIdx(key);

            auto result = intervals[int_idx]->pivot(key);

            shared_ptr<interval> left_part = get<0>(result);
            shared_ptr<interval> right_part = get<1>(result);
            bool found_in_interval = get<2>(result);

            vector<shared_ptr<interval>> left_result = split(left_part, false, n_recursions);
            vector<shared_ptr<interval>> right_result = split(right_part, true, n_recursions);
            vector<shared_ptr<interval>> lesser;

            size_t lesser_reserve = int_idx + left_result.size();
            lesser.reserve(lesser_reserve);

            for (int i = 0; i < int_idx; ++i) lesser.emplace_back(intervals[i]);
            lesser.insert(lesser.end(), left_result.begin(), left_result.end());

            vector<shared_ptr<interval>> greater;
            greater.reserve(right_result.size() + (intervals.size() - int_idx));
            greater.insert(greater.end(), right_result.begin(), right_result.end());
            for (size_t i = int_idx + 1; i < intervals.size(); ++i) greater.emplace_back(intervals[i]);

            return make_tuple(gap(lesser, comp), gap(greater, comp), found_in_interval);
        }

        template <typename Iterator, typename f>
        int perform_merges(Iterator begin, Iterator end, f deletion) {
            int n_out = 0;
            int i = 0;
            auto it = begin;
            for (;;) {
                auto it2 = it; ++it2;
                if (it2 == end) break;
                int cur_size = (int)(*it)->size();
                int next_size = (int)(*it2)->size();
                int n_in = (int)size() - cur_size - n_out;
                if (n_out + cur_size >= n_in - next_size) return i;
                if (n_out >= cur_size + next_size) {
                    (*it)->merge(*it2);
                    deletion(it2);
                } else {
                    n_out += cur_size;
                    ++it; ++i;
                }
            }
            return i;
        }

        void rebalance() {
            list<shared_ptr<interval>> intervals_list(intervals.begin(), intervals.end());
            last_left_idx = perform_merges(intervals_list.begin(), intervals_list.end(),
                                           [&intervals_list](typename list<shared_ptr<interval>>::iterator it){
                                               intervals_list.erase(it);
                                           });
            perform_merges(intervals_list.rbegin(), intervals_list.rend(),
                           [&intervals_list](typename list<shared_ptr<interval>>::reverse_iterator it){
                               intervals_list.erase(next(it).base());
                           });
            intervals.assign(intervals_list.begin(), intervals_list.end());
        }

        bool erase(const T &key) {
            if (intervals.empty()) return false;

            int int_idx = getIntervalIdx(key);

            if (intervals[int_idx]->erase(key)) {
                --gap_size;
                if (intervals[int_idx]->empty()) {
                    intervals.erase(intervals.begin() + int_idx);
                }
                refresh_stats();
                return true;
            }

            for (size_t i = 0; i < intervals.size(); ++i) {
                if (i != int_idx && intervals[i]->erase(key)) {
                    --gap_size;
                    if (intervals[i]->empty()) {
                        intervals.erase(intervals.begin() + i);
                    }
                    refresh_stats();
                    return true;
                }
            }
            return false;
        }

        size_t size() const { return gap_size; }
        bool empty( ) const { return size() == 0; }
        T get_max() const { return gap_max; }
        T get_min() const { return gap_min; }
        void print() { cout << gap_max << endl; }
    };

    struct HybridPriority {
        static double get(const gap& g) {
            int min_k = (int)g.get_min();
            int max_k = (int)g.get_max();

            float total_N = (float)std::max(1UL, lazy_search_tree::total_elements_global);
            float w_dynamic = (float)g.size() / total_N;

            float w_oracle = lazy_search_tree::oracle.get_gap_weight(min_k, max_k, w_dynamic);

            float w_final = w_oracle;
            float THRESHOLD_MULTIPLIER = 10.0f; // Tune this based on model's confidence

            if (w_dynamic > w_oracle * THRESHOLD_MULTIPLIER) {
                w_final = w_dynamic;
            }

            // Clamp w_final to avoid log(0) or negative values
            if (w_final < 1e-9f) w_final = 1e-9f;
            if (w_final > 1.0f) w_final = 1.0f;

            // - floor( log2( log2( 1 / w ) ) ) + U(0,1)
            double inner_log = std::log2(1.0 / (double)w_final);

            if (inner_log < 1.0) inner_log = 1.0;
            double tier = std::floor(std::log2(inner_log));

            double noise = FastRNG::next_uniform();
            return -tier + noise;
        }
    };

    treap<gap, std::less<gap>, HybridPriority> gap_ds;

public:
    lazy_search_tree(int max_key_range = 1000000) : lst_size(0) {
        if (total_elements_global == 0) {
            oracle.load("src/learned/weights.txt");
            oracle.build_cache(0, max_key_range);
        }
    }

    void push(const T &key) { insert(key); }

    void insert(const T &key) {
        total_elements_global++;

        if (empty()) {
            gap r_gap = gap(key, comp);
            r_gap.last_priority_update_size = 1;
            gap_ds.insert(r_gap);
        } else {
            gap& r_gap = gap_ds.lower_bound_or_last(gap(key, comp));
            r_gap.insert(key);

            // Only trigger O(log m) Treap rebalancing if the gap grew by 50%
            if (r_gap.size() > r_gap.last_priority_update_size * 1.5) {
                gap_ds.update_priority(r_gap);
                r_gap.last_priority_update_size = r_gap.size();
            }
        }
        ++lst_size;
    }

    int count(const T &key) {
        if (empty()) return false;

        T min_val = std::numeric_limits<T>::lowest();
        gap query_gap(key, min_val, comp);

        gap &r_gap = gap_ds.lower_bound_or_last(query_gap);

        // Extract membership directly from restructure
        tuple<gap, gap, bool> new_gaps = r_gap.restructure(key, 2);

        gap& left_gap = get<0>(new_gaps);
        gap& right_gap = get<1>(new_gaps);
        bool found = get<2>(new_gaps);

        gap_ds.erase(r_gap);

        if (!right_gap.empty()) { gap_ds.insert(right_gap); }
        if (!left_gap.empty()) { gap_ds.insert(left_gap); }

        return found;
    }

    void erase(const T &key) {
        if (empty()) return;

        T min_val = std::numeric_limits<T>::lowest();
        gap query_gap(key, min_val, comp);

        gap &r_gap = gap_ds.lower_bound_or_last(query_gap);

        // Erase the item in-place. Shrinking bounds never violates the BST.
        if (r_gap.erase(key)) {
            --lst_size;
            if (total_elements_global > 0) {
                --total_elements_global;
            }

            // Safely remove the gap if it hits zero elements
            if (r_gap.empty()) {
                gap_ds.erase(r_gap);
            }
        }
    }

    size_t size( ) const { return lst_size; }
    bool empty( ) const { return size() == 0; }
    void print() { gap_ds.print(); }
};

template<typename T, typename Comp>
LearnedOracle lazy_search_tree<T, Comp>::oracle;

template<typename T, typename Comp>
unsigned long lazy_search_tree<T, Comp>::total_elements_global = 0;

#endif