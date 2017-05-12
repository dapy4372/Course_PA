# ifndef KDTREE_H
# define KDTREE_H

# include "utils.h"
# include "Node.h"
# include <vector>

template < class T > 
class KdTree
{
public:
    KdTree() : _root(NULL), _numNode(0), _nn(NULL){}
    void insert(const Element<T> &);
    void rangeSearch(const T [DIM][2]);
    void deleteNode(const Element<T> &);
    void NNSearch(const Element<T> &);
    void printNNSearch() const;
    void printRangeSearchRes(const T [DIM][2], FILE *) const;

private:
    Node<T> *_root;
    int _numNode;
    Node<T> *_nn;
    std::vector< Node<T> * > _rangeSearchRes;
    T _nndis;

    Node<T> *search(const Element<T> &) const;
    Node<T> *successor(Node<T> *, const unsigned &, const unsigned &) const;
    void rangeSearch(Node<T> *, const unsigned &, const T [DIM][2]);
    void deleteNode(Node<T> *);
    void deleteNode(Node<T> *, const unsigned &);
    void NNSearch(const Element<T> &, Node<T> *, T &);
    //void NNSearch(const Element<T> &, Node<T> *, T &, Node<T> *);
    
    T squaredDistance(const Element<T> &, const Element<T> &) const;
    void printNode(const Node<T> &, FILE *) const;
    void printNotInTree(const Element<T> &) const;
};

# include "kdtree.cpp"

# endif
