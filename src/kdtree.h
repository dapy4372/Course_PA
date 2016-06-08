# ifndef KDTREE_H
# define KDTREE_H
# define DIM 2

# include "utils.h"
# include "Node.h"

template < class T > 
class KdTree
{
public:
    KdTree() : _root(NULL), _numNode(0) {}
    void insert(const Element<T> &);
    void rangeSearch(const T [][DIM]);
    Node<T> *search(const Element<T> &);
    //void delete(const Element<T> &);
    //void nearestNeighborSearch(const Element<T> &);

private:
    Node<T> *_root;
    int _numNode;
    //void search(const Element<T> &);
    void rangeSearch(Node<T> *, const int &, const T [][DIM]);
    //void delete(Node<T> *, const int &);
    Node<T> *smallest(Node<T> const *, const int &, const int &);
};

# endif
