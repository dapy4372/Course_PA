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
    Node();
    Node(const Element<T> &e, const unsigned &level, Node *par, Node *l = NULL, Node *r = NULL) 
    : _el(e), _level(level), _parent(par), _left(l), _right(r) {}
    void print() const;
    unsigned getLevel() const { return _level; }
    void setLevel(const unsigned &i) { _level = i; }
    Element<T> _el;
    unsigned _level;
    Node *_parent, *_left, *_right;
    //Node *getParent() const { return _parent; }
    //void setParent(Node *par) { _parent = par; }
    //Node *getLeft() const { return _left; }
    //void setLeft(Node *l) { _left = l; }
    //Node *getRight() const { return _right; }
    //void setRight(Node *r) { _right = r; }
    //Element<T> *getElement() const { return _el; }
    //void setElement(const Element<T> &el) { _el = el; }
private:
};

# endif
