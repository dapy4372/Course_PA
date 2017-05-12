# ifndef _UTILS_H
# define _UTILS_H

# include <vector>
# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>
# include <sys/times.h>
# include "Node.h"

using namespace std;

template < class T > static vector< Element<T> > readFile( char *filename);
template < class T > static T square(const T &a);
static void err_sys(const char *);
static void printTimes(clock_t real, struct tms *tmsstart, struct tms *tmsend);

# include "utils.cpp"

# endif
