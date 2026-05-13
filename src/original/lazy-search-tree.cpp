// An implementation of the lazy search tree data structure, from the paper "Lazy Search Trees"
// by Bryce Sandlund and Sebastian Wild. A splay tree is used as the data structure for the gaps
// and a linked list of vectors is the data structure for the intervals, which allows O(1) time
// merge, insert, and delete while maintaining the O(min(n, q log n)) pointer bound.

#ifndef LAZY_SEARCH_TREE
#define LAZY_SEARCH_TREE

#define INF 1000000000

#include "splay.cpp"
#include <list>
#include <cstdlib>
#include <memory>
#include <limits>
#include <cstdint>
#include <tuple>

using namespace std;

// Fast Thread-Local RNG
// Replaces rand() to avoid global locks and improve throughput.
struct FastRNG {
    static inline uint32_t next() {
        static thread_local uint32_t x = 123456789;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return x;
    }
};

template<typename T, typename Comp = std::less<T>>
class lazy_search_tree {
private:
  Comp comp;
  size_t lst_size;

  // data structure that contains a set of intervals within a gap.
  class gap {
  private:
    Comp comp;

    // an interval within a gap.
    class interval {
    private:
      Comp comp;
      T max_e;
      T min_e;
      size_t int_size;

      // intervals require a linked list data structure for O(1) merging, but by using a linked list of
      // vectors, we can take advantage of larger built intervals and inserted elements, reducing the
      // number of pointers in the entire data structure to O(min(n, q log n)).
      list<vector<T>> elements;

    public:
      // returns an element uniformly at random from the interval. Time complexity is no worse
      // than linear in the size of the interval, but typically more like logarithmic.
      T sample() {
        if (size() == 0) return T();
        int idx = FastRNG::next() % size();
        for (vector<T>& vec : elements) {
          if (idx < (int)vec.size()) {
            return vec[idx];
          }
          idx -= static_cast<int>(vec.size()); //idx -= vec.size();
        }
        return T();
      }

      // merges 'other' into this interval, destroying 'other'.
      void merge(shared_ptr<interval> other) {
        if (other->int_size == 0) return;
        int_size += other->int_size;

        if (comp(max_e, other->max_e)) max_e = other->max_e;
        if (comp(other->min_e, min_e)) min_e = other->min_e;

        elements.splice(elements.end(), other->elements);
        other->int_size = 0;
      }

      // insert an element into this interval.
      void insert(const T &element) {
        // In-place construction
        elements.front().emplace_back(element);

        if (comp(max_e, element)) max_e = element;
        if (comp(element, min_e)) min_e = element;
        ++int_size;
      }

      // create an interval from a vector of type T.
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

      // create an interval with a single element.
      interval(const T &element, Comp c) : comp(c) {
        int_size = 1;
        elements.emplace_back(vector<T>({element}));
        max_e = element;
        min_e = element;
      }

    tuple<shared_ptr<interval>, shared_ptr<interval>, bool> pivot(const T &p) {
        // Flatten the interval if it's fragmented from prior merges
        if (elements.size() > 1) {
            vector<T> flat;
            flat.reserve(int_size);
            for (auto& vec : elements) {
                // Bulk insert is highly optimized
                flat.insert(flat.end(), std::make_move_iterator(vec.begin()), std::make_move_iterator(vec.end()));
            }
            elements.clear();
            elements.emplace_back(std::move(flat));
        }

        vector<T>& data = elements.front();
        bool found = false;

        // In-place partition.
        // Elements satisfying the lambda (less than pivot) move to the front.
        auto bound = std::partition(data.begin(), data.end(), [&](const T& e) {
            if (!comp(e, p) && !comp(p, e)) {
                found = true;
                // Distribute duplicates evenly using your RNG logic
                return (FastRNG::next() & 1) == 0;
            }
            return comp(e, p);
        });

        // Slice into two new vectors (single block allocation per side)
        vector<T> lesser_vec(std::make_move_iterator(data.begin()), std::make_move_iterator(bound));
        vector<T> greater_vec(std::make_move_iterator(bound), std::make_move_iterator(data.end()));

        return make_tuple(make_shared<interval>(lesser_vec, comp),
                          make_shared<interval>(greater_vec, comp),
                          found);
    }

      // O(1) deletion using "Swap and Pop" for unsorted vectors
      bool erase(const T &key) {
        for (auto list_it = elements.begin(); list_it != elements.end(); ++list_it) {
          auto& vec = *list_it;
          for (auto vec_it = vec.begin(); vec_it != vec.end(); ++vec_it) {

            // Check for exact equality using the comparator
            if (!comp(*vec_it, key) && !comp(key, *vec_it)) {
              std::iter_swap(vec_it, vec.end() - 1);
              vec.pop_back();
              --int_size;

              // Clean up the vector block if it becomes empty
              if (vec.empty()) {
                elements.erase(list_it);
              }

              // Recompute the boundaries if we removed the interval's min or max
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

      // compare gaps to one another via their maximum element.
      bool operator< (const interval& other) const {
        if (comp(max_e, other.max_e)) return true;
        if (comp(other.max_e, max_e)) return false;
        return comp(min_e, other.min_e);
      }

      // return the number of elements in this interval.
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

    // Helper to refresh gap min/max from its intervals after rebalancing
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

    // initialize a gap with a vector of intervals.
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

    // returns smallest interval with maximum element larger than or equal to key.
    // Optimized to provide O(1) average case insert, O(log log Delta_i) worst-case.
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

    // pick the pivot element to split the interval.
    T pick_pivot(int sample_size, shared_ptr<interval> g_int) {
      vector<T> pivots;
      pivots.reserve(sample_size);
      for (int i = 0; i < sample_size; ++i) pivots.push_back(g_int->sample());
      std::sort(pivots.begin(), pivots.end(), comp);
      return pivots[sample_size / 2];
    }

    // split interval g_int, recursing on either the left or right side of the split.
    // based on the value of "recurse_left". Return a vector of all resulting intervals.
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

  public:
    gap(const T &key_max, const T &key_min, Comp c) : comp(c), gap_size(0) {
        gap_max = key_max;
        gap_min = key_min;
    }

    // create a gap with a single interval containing a single element.
    gap(const T &key, Comp c) : comp(c) {
      gap_size = 1;

      intervals.emplace_back(make_shared<interval>(key, c));
      gap_max = key;
      gap_min = key;
    }

    // compare gaps to one another via their maximum element.
    bool operator< (const gap& other) const {
      if (comp(gap_max, other.gap_max)) return true;
      if (comp(other.gap_max, gap_max)) return false;
      return comp(gap_min, other.gap_min);
    }

    // insert key into this gap.
    void insert(const T &key) {
      intervals[getIntervalIdx(key)]->insert(key);
      ++gap_size;
      if (comp(gap_max, key)) gap_max = key;
      if (comp(key, gap_min)) gap_min = key;
    }

    // Returns {left_gap, right_gap, found_boolean}
    tuple<gap, gap, bool> restructure(const T &key, int n_recursions) {
      int int_idx = getIntervalIdx(key);

      auto result = intervals[int_idx]->pivot(key);

      shared_ptr<interval> left_part = get<0>(result);
      shared_ptr<interval> right_part = get<1>(result);
      bool found_in_interval = get<2>(result);

      vector<shared_ptr<interval>> left_result = split(left_part, false, n_recursions);
      vector<shared_ptr<interval>> right_result = split(right_part, true, n_recursions);
      vector<shared_ptr<interval>> lesser;

      // Reserve space for lesser
      lesser.reserve(int_idx + left_result.size());
      for (int i = 0; i < int_idx; ++i) lesser.emplace_back(intervals[i]);
      lesser.insert(lesser.end(), left_result.begin(), left_result.end());

      vector<shared_ptr<interval>> greater;
      // Reserve space for greater
      greater.reserve(right_result.size() + (intervals.size() - int_idx));
      greater.insert(greater.end(), right_result.begin(), right_result.end());
      for (size_t i = int_idx + 1; i < intervals.size(); ++i) greater.emplace_back(intervals[i]);

      return make_tuple(gap(lesser, comp), gap(greater, comp), found_in_interval);
    }

    // Merges adjacent intervals if they are small enough, maintaining the pointer bound.
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

    // rebalance according to (A) and (B).
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

      // Try the designated interval first
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

    // return the number of elements in this gap.
    size_t size() const { return gap_size; }
    // return if this gap is empty.
    bool empty( ) const { return size() == 0; }
    T get_max() { return gap_max; }
    void print() { cout << gap_max << endl; }
  };

  splay_tree<gap> gap_ds;

public:
  lazy_search_tree() : lst_size(0) {}

  void push(const T &key) { insert(key); }

  // insert key into the lazy search tree.
  void insert(const T &key) {
    if (empty()) {
      gap r_gap = gap(key, comp);
      gap_ds.insert(r_gap);
    } else {
      gap& r_gap = gap_ds.lower_bound_or_last(gap(key, comp));
      r_gap.insert(key);
    }
    ++lst_size;
  }

  // return if key is present in the lazy search tree and restructure
  bool count(const T &key) {
    if (empty()) {
      return false;
    } else {
      T min_val = std::numeric_limits<T>::lowest();
      gap query_gap(key, min_val, comp);

      gap &r_gap = gap_ds.lower_bound_or_last(query_gap);

      // Extract membership directly from restructure
      tuple<gap, gap, bool> new_gaps = r_gap.restructure(key, 2);

      gap& left_gap = get<0>(new_gaps);
      gap& right_gap = get<1>(new_gaps);
      bool found = get<2>(new_gaps);

      gap_ds.erase(r_gap);

      if (!right_gap.empty()) {
        gap_ds.insert(right_gap);
      }
      if (!left_gap.empty()) {
        gap_ds.insert(left_gap);
      }

      return found;
    }
  }

  void erase(const T &key) {
    if (empty()) return;

    T min_val = std::numeric_limits<T>::lowest();
    gap query_gap(key, min_val, comp);

    gap &r_gap = gap_ds.lower_bound_or_last(query_gap);

    // Erase the item in-place. Shrinking bounds never violates the BST.
    if (r_gap.erase(key)) {
        --lst_size;

        if (r_gap.empty()) {
            gap_ds.erase(r_gap);
        }
    }
  }

  void print() { gap_ds.print(); }
  size_t size( ) const { return lst_size; }
  bool empty( ) const { return size() == 0; }
};

#endif
