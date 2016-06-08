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
        prev -> left = new Node<T>(el);
    else
        prev -> right = new Node<T>(el);
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
    for(int j = 0; j < _numNode - 1; ++j){
        if(!(ranges[j][0] <= p->el.keys[j] && p->el.keys[j] <= ranges[j][1]))
            found = false;
            break;
    }
    if(found)
        p->print();
    if(p->left != NULL && ranges[i][0] <= p->el.keys[i])
        rangeSearch(p->left, (i + 1) % DIM, ranges);
    if(p->right != NULL && p->el.keys[i] <= ranges[i][1])
        rangeSearch(p->right, (i + 1) % DIM, ranges);
}

template < class T >


//void kdTree<T>::delete(const Element<T> &)
//{


//}
//void rangeSearch(const int &, const int &, const int &, const int &);
//void nearestNeighborSearch(const Element<T> &);
