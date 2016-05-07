# ifndef KDTREE_H
# define KDTREE_H
# define DIM 2

# include <stddef.h>
# include "utils.h"

template < class T > 
class KdTree
{
public:
    KdTree() : _root(NULL) {}
    void insert(const Element<T> &el);

private:
    BSTNode<T> *_root;
};

# endif
