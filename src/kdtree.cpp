# include "utils.h"
# include <stdlib.h>
# include <stdio.h>
# include <limits>
# include <math.h>

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
void KdTree<T>::rangeSearch(const T ranges[DIM][2])
{
    if( !_rangeSearchRes.empty() )
        _rangeSearchRes.clear();

    if( _root != NULL )
        rangeSearch(_root, 0, ranges);
}

template < class T >
void KdTree<T>::rangeSearch(Node<T> *p, const unsigned &i, const T ranges[][DIM])
{
    bool found = true;
    for(int j = 0; j < DIM; ++j) {
        if( !(ranges[j][0] <= p->_el.keys[j] && p->_el.keys[j] <= ranges[j][1]) ){
            found = false;
            break;
        }
    }
    if( found )
        _rangeSearchRes.push_back(p);
        //p->print();
    if( p->_left != NULL && ranges[i][0] <= p->_el.keys[i] )
        rangeSearch(p->_left, (i + 1) % DIM, ranges);
    if( p->_right != NULL && p->_el.keys[i] <= ranges[i][1] )
        rangeSearch(p->_right, (i + 1) % DIM, ranges);
}

template < class T >
Node<T> *KdTree<T>::successor(Node<T> *q, const unsigned &i, const unsigned &j) const
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
        Node<T> *l_smallest = successor(q->_left, i, (j+1) % DIM);
        if( smallest_node->_el.keys[i] >= l_smallest->_el.keys[i] )
            smallest_node = l_smallest;
    }
    // Check right subtree smallest Node
    if( q->_right != NULL ) {
        Node<T> *r_smallest = successor(q->_right, i, (j+1) % DIM);
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
    else
        printNotInTree(el);
}

template < class T >
void KdTree<T>::deleteNode(Node<T> *p)
{
    if( p->_parent->_right == p )
        p->_parent->_right = NULL;
    else
        p->_parent->_left = NULL;
    delete p;
    --_numNode;
}

template < class T >
void KdTree<T>::deleteNode(Node<T> *p, const unsigned &level)
{
    if( p->_left == NULL && p->_right == NULL )    // p is leaf
        deleteNode(p);
    else {
        Node<T> *q;
        if( p->_right != NULL )    // if p have right subtree, find smallest in right subtree
            q = successor( p->_right, level, (level + 1) % DIM );
        else {    // if p does not have right subtree, find smallest in left subtree and swap the left subtree to right subtree
            q = successor(p->_left, level, (level + 1) % DIM);
            if( q != NULL ) {
                p->_right = p->_left;
                p->_left = NULL;
            }
        }
        p->_el = q->_el;
        deleteNode(q, level);
    }
}

template < class T >
T KdTree<T>::squaredDistance(const Element<T> &el1, const Element<T> &el2) const
{
    return square(el1.keys[0] - el2.keys[0]) + square(el1.keys[1] - el2.keys[1]);
}

template < class T >
void KdTree<T>::NNSearch(const Element<T> &query)
{   
    _nndis = std::numeric_limits<T>::infinity();
    //Node<T> nn(query, 1, NULL);
    NNSearch(query, _root, _nndis);
    //NNSearch(query, _root, _nndis, _nn);
    //printNode(nn, stdout);
    //fprintf(stdout, "\nThe distance is %lf.\n\n", dis);
}

template < class T >
//void KdTree<T>::NNSearch(const Element<T> &query, Node<T> *refer, T &nearst_squared_dis, Node<T> &nn)
void KdTree<T>::NNSearch(const Element<T> &query, Node<T> *refer, T &nearst_squared_dis)
{
    T query2refer_dis = squaredDistance(refer->_el, query);
    if( query2refer_dis < nearst_squared_dis ) {
        nearst_squared_dis = query2refer_dis;
        _nn = refer;
    }
      
    T query2split_axis = query.keys[refer->getLevel()] - refer->_el.keys[refer->getLevel()];
    // trim traverse path
    bool trim_left = false;
    bool trim_right = false;
    if( nearst_squared_dis < square(query2split_axis) )
        query2split_axis < 0 ? trim_right = true : trim_left = true;

    if( !trim_left && (refer->_left != NULL) )
        NNSearch(query, refer->_left, nearst_squared_dis);
//        NNSearch(query, refer->_left, nearst_squared_dis, nn);
    if( !trim_right && (refer->_right != NULL) )
        NNSearch(query, refer->_right, nearst_squared_dis);
//        NNSearch(query, refer->_right, nearst_squared_dis, nn);
};

template < class T >
void KdTree<T>::printNode(const Node<T> &n, FILE *fp) const
{
    fprintf(fp, "    (%lf, %lf)", (n._el).keys[0], (n._el).keys[1]);
}

template < class T >
void KdTree<T>::printNotInTree(const Element<T> &el) const
{
    fprintf(stderr, "    The node (%lf, %lf) is not in the tree!\n", el.keys[0], el.keys[1]);
    exit(EXIT_FAILURE);
}

template < class T >
void KdTree<T>::printRangeSearchRes(const T ranges[DIM][2], FILE *fp) const
{
    fprintf(stdout, "    The given rectangle is (%lf, %lf), (%lf, %lf), (%lf, %lf), (%lf, %lf).\n", ranges[0][0], ranges[1][0], ranges[0][0], ranges[1][1], ranges[0][1], ranges[1][0], ranges[0][1], ranges[1][1]);
    if( !_rangeSearchRes.empty() ) {
        for( unsigned i = 0; i < _rangeSearchRes.size(); ++i ){
            printNode(*_rangeSearchRes[i], fp);
            if( (i + 1) % 5 == 0)
                fprintf(fp, "\n");
        }
        fprintf(stdout, "    There are %lu node in the given range.\n", _rangeSearchRes.size() );
    }
    else
        fprintf(stdout, "    There is not any node in the given range.\n");
}

template < class T >
void KdTree<T>::printNNSearch() const
{
    printNode(*_nn, stdout);
    fprintf(stdout, "\n    The distance is %lf.\n", sqrt(_nndis));
}
template class KdTree<double>;
