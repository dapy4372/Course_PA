# include "kdtree.h"
# define DIM 2

template < class T >
void KdTree<T>::insert(const Element<T> &el)
{
    unsigned level = 0;
    BSTNode<T> *p = _root;
    BSTNode<T> *prev = NULL;
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
        _root = new BSTNode<T>(el);
    else if( el.keys[ (level - 1) % DIM ] < prev->el.keys[ (level - 1) % DIM ] )
        prev -> left = new BSTNode<T>(el);
    else
        prev -> right = new BSTNode<T>(el);
};
