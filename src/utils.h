# include "stdio.h"
# include "kdtree.h"
# include <vector>

# define BUF_SIZE 128
//template < class T >
//using std::vector< Element<T> > = typename std::vector< Element<T> >;

// read file
template < class T >
std::vector< Element<T> > readFile( char *filename )
{
    FILE *fp = fopen( filename, "r");
    if( fp == NULL ){
        fprintf(stderr, "open failure!");
        return 1;
    }

    char buf[BUF_SIZE];
    std::vector< Element<T> > el_vec;
    Element<T> el;
    while( fgets(buf, sizeof(buf), fp) ){
        fscanf( fp, "(%d, %d)", &el.keys[0], &el.keys[1] );
        el_vec.push_back(el);
    }
    fclose(fp);
}



