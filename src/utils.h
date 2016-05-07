# include <stdio.h>
# include <stdlib.h>
# include <queue>

# ifndef UTILS_H
# define UTILS_H

using namespace std; 

// TODO: typedef std::vector< Element<T> > 
// template < class T >
// using std::vector< Element<T> > = typename std::vector< Element<T> >;
// read file

template < typename T >
struct Element
{
    T keys[DIM];
};    

template < typename T >
class BSTNode
{
public:
    BSTNode();
    BSTNode(const Element<T> &e, BSTNode *l = NULL, BSTNode *r = NULL) : el(e), left(l), right(r){}
    Element<T> el;
    BSTNode *left, *right;
};

template < typename T >
queue< Element<T> > readFile( char *filename)
{
    FILE *fp = fopen( filename, "r");
    if( fp == NULL ){
        fprintf(stderr, "open failure!");
        exit(1);
    }

    char *line = NULL;
    size_t len = 0;
    Element<T> el;
    queue< Element<T> > el_que;
    while( getline(&line, &len, fp) != -1){
        sscanf(line, "(%f, %f)", &el.keys[0], &el.keys[1] );
        el_que.push(el);
    }
    fclose(fp);
    return el_que;
}

# endif
