# ifndef NODE_H
# define NODE_H
# define DIM 2

# include <stddef.h>

template < class T >
struct Element
{
    T keys[DIM];
};    

template < class T >
class Node
{
public:
    Node();
    Node(const Element<T> &e, Node *l = NULL, Node *r = NULL) : el(e), left(l), right(r){}
    Element<T> el;
    Node *left, *right;
    void print();
};

# endif
