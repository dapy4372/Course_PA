# include <stdio.h>
# include <stdlib.h>
# include <vector>
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
vector< Element<T> > readFile( char *filename)
{
    FILE *fp = fopen( filename, "r");
    if( fp == NULL ){
        fprintf(stderr, "open failure!");
        exit(1);
    }

    char *line = NULL;
    size_t len = 0;
    Element<T> el;
    vector< Element<T> > el_vec;
    while( getline(&line, &len, fp) != -1){
        sscanf(line, "%lf %lf", &el.keys[0], &el.keys[1] );
        el_vec.push_back(el);
    }
    fclose(fp);
    return el_vec;
}

template < class T >
T square(const T &a)
{
    return a*a;
}

# endif
