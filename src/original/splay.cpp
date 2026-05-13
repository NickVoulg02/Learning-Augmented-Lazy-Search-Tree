#ifndef SPLAY_TREE
#define SPLAY_TREE

#include <functional>
#include <iostream>
#include <vector>
#include <algorithm>

template<typename T, typename Comp = std::less<T>>
class splay_tree {
private:
  Comp comp;
  size_t p_size;

  struct node {
    node *left, *right;
    node *parent;
    T key;
    node( const T& init = T( ) ) : left( nullptr ), right( nullptr ), parent( nullptr ), key( init ) { }
    ~node( ) { }
  } *root;

  // Rotations mostly taken from wikipedia
  void left_rotate( node *x ) {
    node *y = x->right;
    if(y) {
      x->right = y->left;
      if( y->left ) y->left->parent = x;
      y->parent = x->parent;
    }
    if( !x->parent ) root = y;
    else if( x == x->parent->left ) x->parent->left = y;
    else x->parent->right = y;
    if(y) y->left = x;
    x->parent = y;
  }

  void right_rotate( node *x ) {
    node *y = x->left;
    if(y) {
      x->left = y->right;
      if( y->right ) y->right->parent = x;
      y->parent = x->parent;
    }
    if( !x->parent ) root = y;
    else if( x == x->parent->left ) x->parent->left = y;
    else x->parent->right = y;
    if(y) y->right = x;
    x->parent = y;
  }

  // Moves node x all the way to the root using specific rotation patterns
  void splay( node *x ) {
    while( x->parent ) {
      if( !x->parent->parent ) {
        if( x->parent->left == x ) right_rotate( x->parent );   // Zig
        else left_rotate( x->parent );
      } else if( x->parent->left == x && x->parent->parent->left == x->parent ) {
        right_rotate( x->parent->parent );  // Zig-Zig
        right_rotate( x->parent );
      } else if( x->parent->right == x && x->parent->parent->right == x->parent ) {
        left_rotate( x->parent->parent );
        left_rotate( x->parent );
      } else if( x->parent->left == x && x->parent->parent->right == x->parent ) {
        right_rotate( x->parent );  // Zig-Zag
        left_rotate( x->parent );
      } else {
        left_rotate( x->parent );
        right_rotate( x->parent );
      }
    }
  }

  // Replaces the subtree rooted at u with the subtree rooted at v by updating u's parent to point to v.
  void replace( node *u, node *v ) {
    if( !u->parent ) root = v;
    else if( u == u->parent->left ) u->parent->left = v;
    else u->parent->right = v;
    if( v ) v->parent = u->parent;
  }

  node* subtree_minimum( node *u ) {
    while( u->left ) u = u->left;
    return u;
  }

  node* subtree_maximum( node *u ) {
    while( u->right ) u = u->right;
    return u;
  }

  // Returns the smallest node that compares >= key.
  // If we find an exact match, we continue searching LEFT.
  // This ensures that if multiple duplicates keys exist, we return
  // the "first" (left-most) one, which is required for correct Lazy Tree lookups.
  node* find_or_successor(const T &key) {
    node *z = root;
    node *last = nullptr;
    node *ret = nullptr;

    while (z) {
      last = z;
      if (comp(z->key, key)) {
          // z < key -> Go Right
          z = z->right;
      } else {
          // z >= key -> Potential Successor found.
          // Record it, but keep searching LEFT for an earlier duplicate.
          ret = z;
          z = z->left;
      }
    }

    if (ret) {
        splay(ret);
        return ret;
    }
    if (last) splay(last);
    return last;
  }

  // Returns the node containing key, if such a node exists, and returns null otherwise.
  node* find(const T &key) {
    node *z = root;
    while(z) {
        if (comp(z->key, key)) z = z->right;
        else if (comp(key, z->key)) z = z->left;
        else {
            splay(z);
            return z;
        }
    }
    return nullptr;
  }

public:
  splay_tree( ) : root( nullptr ), p_size( 0 ) { }

  // Iterative Destructor to prevent stack overflow on large trees
  ~splay_tree() {
    if (!root) return;
    std::vector<node*> stack;
    stack.reserve(p_size > 0 ? (p_size < 1000000 ? p_size : 1000000) : 100);
    stack.push_back(root);
    while (!stack.empty()) {
        node* curr = stack.back();
        stack.pop_back();
        if (curr->left) stack.push_back(curr->left);
        if (curr->right) stack.push_back(curr->right);
        delete curr;
    }
  }

  void insert( const T &key ) {
    node *z = root;
    node *p = nullptr;

    while( z ) {
      p = z;
      // Right-Biased Insertion:
      // If key >= z->key, we go Right.
      // This places duplicates to the Right, preserving the order required
      // for the Left-biased 'find_or_successor' to work correctly.
      if( comp( key, z->key ) ) z = z->left;
      else z = z->right;
    }

    z = new node( key );
    z->parent = p;

    if( !p ) root = z;
    else if( comp( key, p->key ) ) p->left = z;
    else p->right = z;

    splay( z );
    p_size++;
  }

  void erase( const T &key ) {
    node *z = find( key );
    if( !z ) return;

    if( !z->left ) replace( z, z->right );
    else if( !z->right ) replace( z, z->left );
    else {
      node *y = subtree_minimum( z->right );
      if( y->parent != z ) {
        replace( y, y->right );
        y->right = z->right;
        y->right->parent = y;
      }
      replace( z, y );
      y->left = z->left;
      y->left->parent = y;
    }

    delete z;
    p_size--;
  }

  bool count(const T &key) {
    return find(key) != nullptr;
  }

  // Returns the smallest key that compares >= key, or the largest node
  // if no other node exists.
  T& lower_bound_or_last(const T &key) {
    node *ret = find_or_successor(key);
    return ret->key;
  }

  const T& minimum( ) { return subtree_minimum( root )->key; }
  const T& maximum( ) { return subtree_maximum( root )->key; }
  bool empty( ) const { return root == nullptr; }
  size_t size( ) const { return p_size; }
};
#endif
