# ifndef KDTREE_H
# define KDTREE_H
# define DIM 2

template< class T >
struct Element
{
    T keys[DIM];
};    

template< class T >
class BSTNode
{
public:
    BSTNode();
    BSTNode(const Element<T> &e, BSTNode *l = NULL, BSTNode *r = NULL);
    Element<T> el;
    BSTNode *left, *right;
};

template < class T >
class KdTree
{
public:
    KdTree() : _root(NULL) {};
    void insert(const Element<T> &);
private:
    Element<T> *_root;
};

# endif
