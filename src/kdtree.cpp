#include "kdtree.h"
# define DIM 2

template < class T >
void KdTree<T>::insert(const Element<T> &el)
{
    unsigned level = 0;
    Node<T> *p = _root;
    Node<T> *prev = NULL;
    // find postion to insert
    while( p != NULL ) {
        prev = p;
        if( el.keys[level] < p->el.keys[level] )
            p = p->left;
        else
            p = p->right;
        level = (level + 1) % DIM;
    }

    // insert the node to the position
    if( _root == NULL )
        _root = new Node<T>(el);
    else if( el.keys[ (level - 1) % DIM ] < prev->el.keys[ (level - 1) % DIM ] )
        prev->left = new Node<T>(el);
    else
        prev->right = new Node<T>(el);
    ++_numNode;
};

template < class T >
void KdTree<T>::rangeSearch(const T ranges[][DIM])
{
    if(_root != NULL)
        rangeSearch(_root, 0, ranges);
}

template < class T >
void KdTree<T>::rangeSearch(Node<T> *p, const int &i, const T ranges[][DIM])
{
    bool found = true;
    for(int j = 0; j < DIM; ++j) {
        if(!(ranges[j][0] <= p->el.keys[j] && p->el.keys[j] <= ranges[j][1])){
            found = false;
            break;
        }
    }
    if(found)
        p->print();
    if(p->left != NULL && ranges[i][0] <= p->el.keys[i])
        rangeSearch(p->left, (i + 1) % DIM, ranges);
    if(p->right != NULL && p->el.keys[i] <= ranges[i][1])
        rangeSearch(p->right, (i + 1) % DIM, ranges);
}

template < class T >
Node<T> *KdTree<T>::smallest(Node<T> const *q, const int &i, const int &j)
{
    Node<T> *tmp = q;
    // Not for delete
    if(i == j) {
        if(q->left != NULL)
            tmp = q = q->left;   // a = b = c equals to tmp = c; a = tmp; b = tmp;
        else
            return q;
    }
    // Check left subtree smallest Node
    if(q->left != NULL) {
        Node<T> *l_smallest = smallest(q->left, i, (j+1) % DIM);
        if(tmp->el.keys[i] >= l_smallest->el.keys[i])
            tmp = l_smallest;
    }
    // Check right subtree smallest Node
    if(q->right != NULL) {
        Node<T> *r_smallest = smallest(q->left, i, (j+1) % DIM);
        if(tmp->el.keys[i] >= r_smallest->el.keys[i])
            tmp = r_smallest;
    }
    return tmp;
}

template < class T >
Node<T> *KdTree<T>::search(const Element<T> &el)
{
    unsigned level = 0;
    Node<T> *p = _root;
    // similar to finding postion to insert
    while( p != NULL ) {
        bool found = true;
        for(int i = 0; i < DIM; ++i){
            if( el.keys[i] != p->el.keys[i] ){
                found = false; 
                break;
            }
        }
        if( found )
            return p;
        if( el.keys[level] < p->el.keys[level] )
            p = p->left;
        else
            p = p->right;
        level = (level + 1) % DIM;
    }
}

//template < class T >



//}
//void kdTree<T>::delete(const Element<T> &)
//{


//}
//void rangeSearch(const int &, const int &, const int &, const int &);
//void nearestNeighborSearch(const Element<T> &);
