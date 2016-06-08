# ifndef KDTREE_H
# define KDTREE_H

# include "utils.h"
# include "Node.h"

template < class T > 
class KdTree
{
public:
    KdTree() : _root(NULL), _numNode(0) {}
    void insert(const Element<T> &);
    void rangeSearch(const T [][DIM]) const;
    Node<T> *search(const Element<T> &) const;
    Node<T> *smallest(Node<T> *, const unsigned &, const unsigned &) const;
    void deleteNode(const Element<T> &);
    //void nearestNeighborSearch(const Element<T> &);

private:
    Node<T> *_root;
    int _numNode;
    void rangeSearch(Node<T> *, const unsigned &, const T [][DIM]) const;
    void deleteNode(Node<T> *);
    void deleteNode(Node<T> *, const unsigned &);
};

# endif
