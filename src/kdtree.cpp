#include "kdtree.h"

template < class T >
void KdTree<T>::insert(const Element<T> &el)
{
    unsigned level = 0;
    Node<T> *p = _root;
    Node<T> *prev = NULL;
    // find postion to insert
    while( p != NULL ) {
        prev = p;
        if( el.keys[level] < p->_el.keys[level] )
            p = p->_left;
        else
            p = p->_right;
        level = (level + 1) % DIM;
    }

    // insert the node to the position
    if( _root == NULL )
        _root = new Node<T>(el, level, NULL);
    else if( el.keys[ (level - 1) % DIM ] < prev->_el.keys[ (level - 1) % DIM ] )
        prev->_left = new Node<T>(el, level, prev);
    else
        prev->_right = new Node<T>(el, level, prev);
    ++_numNode;
}

template < class T >
void KdTree<T>::rangeSearch(const T ranges[][DIM]) const
{
    if( _root != NULL )
        rangeSearch(_root, 0, ranges);
}

template < class T >
void KdTree<T>::rangeSearch(Node<T> *p, const unsigned &i, const T ranges[][DIM]) const
{
    bool found = true;
    for(int j = 0; j < DIM; ++j) {
        if( !(ranges[j][0] <= p->_el.keys[j] && p->_el.keys[j] <= ranges[j][1]) ){
            found = false;
            break;
        }
    }
    if( found )
        p->print();
    if( p->_left != NULL && ranges[i][0] <= p->_el.keys[i] )
        rangeSearch(p->_left, (i + 1) % DIM, ranges);
    if( p->_right != NULL && p->_el.keys[i] <= ranges[i][1] )
        rangeSearch(p->_right, (i + 1) % DIM, ranges);
}

template < class T >
Node<T> *KdTree<T>::smallest(Node<T> *q, const unsigned &i, const unsigned &j) const
{
    Node<T> *smallest_node = q;

    if( i == j ) {
        if( q->_left != NULL )
            smallest_node = q = q->_left;   // a = b = c equals to tmp = c; a = tmp; b = tmp;
        else
            return q;
    }
    // Check left subtree smallest Node
    if( q->_left != NULL ) {
        Node<T> *l_smallest = smallest(q->_left, i, (j+1) % DIM);
        if( smallest_node->_el.keys[i] >= l_smallest->_el.keys[i] )
            smallest_node = l_smallest;
    }
    // Check right subtree smallest Node
    if( q->_right != NULL ) {
        Node<T> *r_smallest = smallest(q->_right, i, (j+1) % DIM);
        if( smallest_node->_el.keys[i] >= r_smallest->_el.keys[i] )
            smallest_node = r_smallest;
    }
    return smallest_node;
}

template < class T >
Node<T> *KdTree<T>::search(const Element<T> &el) const
{
    unsigned level = 0;
    Node<T> *p = _root;
    // similar to finding postion to insert
    while( p != NULL ) {
        bool found = true;
        for(int i = 0; i < DIM; ++i){
            if( el.keys[i] != p->_el.keys[i] ){
                found = false; 
                break;
            }
        }
        if( found )
            return p;
        if( el.keys[level] < p->_el.keys[level] )
            p = p->_left;
        else
            p = p->_right;
        level = (level + 1) % DIM;
    }
    return NULL;
}

template < class T >
void KdTree<T>::deleteNode(const Element<T> &el)
{
    Node<T> *p = search(el);
    if( p != NULL )
        deleteNode(p, p->getLevel());
}

template < class T >
void KdTree<T>::deleteNode(Node<T> *p)
{
    if( p->_parent->_right == p )
        p->_parent->_right = NULL;
    else
        p->_parent->_left = NULL;
    delete p;
}

template < class T >
void KdTree<T>::deleteNode(Node<T> *p, const unsigned &level)
{
    if( p->_left == NULL && p->_right == NULL )    // p is leaf
        deleteNode(p);
    else {
        Node<T> *q;
        if( p->_right != NULL )    // if p have right subtree, find smallest in right subtree
            q = smallest( p->_right, level, (level + 1) % DIM );
        else {    // if p does not have right subtree, find smallest in left subtree and swap the left subtree to right subtree
            q = smallest(p->_left, level, (level + 1) % DIM);
            if( q != NULL ) {
                p->_right = p->_left;
                p->_left = NULL;
            }
        }
        p->_el = q->_el;
        deleteNode(q, level);
    }
}

//void nearestNeighborSearch(const Element<T> &);
