# include "kdtree.h"

KdTree<T>::BSTNode(const Element<T> &e, BSTNode *l = NULL, BSTNode *r = NULL)
{
    el = e;
    left = l;
    right = r;
}

KdTree<T>::insert( const Element &el ) 
{
    unsigned level = 0;
    Element *p = _root;
    Elemnet *prev = NULL;
    while( p != NULL ) {
        prev = p;
        if( el.keys[level] < p->el.keys[level] )
            p = p->left;
        else
            p = p->right;
        level = (level + 1) % DIM;
    }
    if( root == NULL )
        root = new BSTNode(el);
    else if( el.keys[ (level - 1) % DIM ] < p->el.keys[ (level - 1) % DIM ] )
        prev -> left = new BSTNode(el);
    else
        prev -> right = new BSTNode(el);
}
