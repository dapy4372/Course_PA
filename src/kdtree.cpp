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
    ++_numNode;
};

template < class T >
void KdTree<T>::rangeSearch(const T ranges[][DIM])
{
    if(_root != NULL)
        rangeSearch(_root, 0, ranges);
}

template < class T >
void KdTree<T>::rangeSearch(BSTNode<T> *p, const int &i, const T ranges[][DIM])
{
    bool found = true;
    for(int j = 0; j < _numNode - 1; ++j){
        if(!(ranges[j][0] <= p->el.keys[j] && p->el.keys[j] <= ranges[j][i]))
            found = false;
            break;
    }
    if(found)
        p->print();
    if(p->left != NULL && ranges[i][0] <= p->el.keys[0])
        rangeSearch(p->left, (i + 1) % DIM, ranges);
    if(p->right != NULL && ranges[i][1] <= p->el.keys[1])
        rangeSearch(p->right, (i + 1) % DIM, ranges);
}

//void kdTree<T>::delete(const Element<T> &)
//{


//}
//void rangeSearch(const int &, const int &, const int &, const int &);
//void nearestNeighborSearch(const Element<T> &);
