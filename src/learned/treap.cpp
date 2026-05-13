#ifndef TREAP_H
#define TREAP_H

#include <iostream>
#include <functional>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>

struct FastRNG {
    static inline uint32_t next() {
        static thread_local uint32_t x = 123456789;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return x;
    }

    static inline double next_uniform() {
        return next() * 2.3283064365386963e-10;
    }
};

// Random Priority (Standard Treap)
template <typename T>
struct RandomPriority {
    static double get(const T&) {
        return FastRNG::next_uniform();
    }
};


template<typename T, typename Comp = std::less<T>, typename PrioPolicy = RandomPriority<T>>
class treap {
private:
    struct node {
        T key;
        double priority;
        node *left, *right, *parent;

        node(const T& k) : key(k), priority(PrioPolicy::get(k)), left(nullptr), right(nullptr), parent(nullptr) {}
    } *root;

    Comp comp;
    unsigned long t_size;

    void rotate_right(node* y) {
        node* x = y->left;
        if (!x) return;
        y->left = x->right;
        if (x->right) x->right->parent = y;
        x->parent = y->parent;
        if (!y->parent) root = x;
        else if (y == y->parent->left) y->parent->left = x;
        else y->parent->right = x;
        x->right = y;
        y->parent = x;
    }

    void rotate_left(node* x) {
        node* y = x->right;
        if (!y) return;
        x->right = y->left;
        if (y->left) y->left->parent = x;
        y->parent = x->parent;
        if (!x->parent) root = y;
        else if (x == x->parent->left) x->parent->left = y;
        else x->parent->right = y;
        y->left = x;
        x->parent = y;
    }

    void bubble_up(node* x) {
        while (x->parent && x->priority > x->parent->priority) {
            if (x == x->parent->left) rotate_right(x->parent);
            else rotate_left(x->parent);
        }
    }

    void push_down(node* x) {
        while (true) {
            node* largest = x;
            if (x->left && x->left->priority > largest->priority) largest = x->left;
            if (x->right && x->right->priority > largest->priority) largest = x->right;
            if (largest == x) break;
            if (largest == x->left) rotate_right(x);
            else rotate_left(x);
        }
    }

    node* subtree_minimum(node* u) {
        while (u && u->left) u = u->left;
        return u;
    }

    node* subtree_maximum(node* u) {
        while (u && u->right) u = u->right;
        return u;
    }

    // Returns the smallest node >= key.
    // Handles duplicates by continuing to search LEFT after finding a match.
    node* find_or_successor(const T& key) {
        node* z = root;
        node* ret = nullptr;
        while (z) {
            if (comp(z->key, key)) {
                z = z->right;
            } else {
                ret = z;
                z = z->left;
            }
        }
        return ret;
    }

    node* find(const T& key) {
        node* z = root;
        while (z) {
            if (comp(z->key, key)) z = z->right;
            else if (comp(key, z->key)) z = z->left;
            else return z;
        }
        return nullptr;
    }

public:
    treap() : root(nullptr), t_size(0) {}

    ~treap() {
        if (!root) return;
        std::vector<node*> stack;
        stack.reserve(t_size > 0 ? t_size : 100);
        stack.push_back(root);
        while (!stack.empty()) {
            node* curr = stack.back();
            stack.pop_back();
            if (curr->left) stack.push_back(curr->left);
            if (curr->right) stack.push_back(curr->right);
            delete curr;
        }
    }

    void insert(const T& key) {
        node* z = new node(key);
        if (!root) {
            root = z;
            t_size++;
            return;
        }
        node* x = root;
        node* p = nullptr;
        while (x) {
            p = x;
            // Right-Biased: Identical keys go to the RIGHT
            if (comp(x->key, key)) x = x->right;
            else x = x->left;
        }
        z->parent = p;
        if (comp(p->key, z->key)) p->right = z;
        else p->left = z;
        t_size++;
        bubble_up(z);
    }

    bool count(const T& key) {
        return find(key) != nullptr;
    }

    T& lower_bound_or_last(const T& key) {
        node* ret = find_or_successor(key);
        if (ret) return ret->key;

        node* max_node = subtree_maximum(root);
        // Caller must ensure tree is not empty or handle potential crash here if empty
        return max_node->key;
    }

    // Recalculate priority when gap size changes
    void update_priority(const T& key) {
        node* x = find(key);
        if (!x) return;

        double old_p = x->priority;
        x->priority = PrioPolicy::get(x->key);

        if (x->priority > old_p) bubble_up(x);
        else push_down(x);
    }

    void erase(const T& key) {
        node* x = find(key);
        if (!x) return;

        x->priority = -1e18;    // Force sink
        push_down(x);

        if (!x->parent) root = nullptr;
        else if (x == x->parent->left) x->parent->left = nullptr;
        else x->parent->right = nullptr;

        delete x;
        t_size--;
    }

    unsigned long size() const { return t_size; }
    bool empty() const { return root == nullptr; }
    void print() { }
};

#endif
