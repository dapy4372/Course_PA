# include <stdio.h>
# include <stdlib.h>
# include <queue>
# include <iostream>
# include "Node.h"

# ifndef UTILS_H
# define UTILS_H

using namespace std; 

// TODO: typedef std::vector< Element<T> > 
// template < class T >
// using std::vector< Element<T> > = class std::vector< Element<T> >;
// read file

template < class T >
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
        sscanf(line, "%f %f", &el.keys[0], &el.keys[1] );
        el_que.push(el);
    }
    fclose(fp);
    return el_que;
}

# endif
