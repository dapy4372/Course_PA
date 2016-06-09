# ifndef NODE_H
# define NODE_H
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
    Node<T>();
    Node<T>(const Element<T> &e, const unsigned &level, Node *par, Node *l = NULL, Node *r = NULL) 
    : _el(e), _level(level), _parent(par), _left(l), _right(r) {}
    void print() const;
    unsigned getLevel() const { return _level; }
    Element<T> _el;
    unsigned _level;
    Node *_parent, *_left, *_right;
};

# endif
