# include <iostream>
# include <vector>
# include <stdlib.h>
# include <math.h>
# include <stdio.h>
# include <sys/times.h>

# include "kdtree.h"
# include "Node.h"
# include "utils.h"
# define FOR_NNPERSEC 10000
using namespace std;

int main(int argc, char *argv[])
{
    if(argc != 2) {
        cout << "Usage: \n"
             << "      " << argv[0] << " <input file>\n\n"
             << "Example: \n"
             << "      " << argv[0] << " ./data/input1.txt\n\n";
        exit(0);
    }

    // read file
    vector< Element<double> > el_vec = readFile<double>( argv[1] );

    struct tms tmsstart, tmsend;
    clock_t start, end;
    if( (start = times(&tmsstart)) == -1 )
        err_sys("time error");

    // build kdtree
    KdTree<double> myKdtree;
    for( int i = el_vec.size() - 1; i >= 0; --i )
        myKdtree.insert(el_vec[i]);

    if( (end = times(&tmsend)) == -1 )
        err_sys("time error");
    fprintf(stdout, "The time to build a kd tree:\n");
    printTimes(end - start, &tmsstart, &tmsend);

    /** (a) Find the nearest neighbor of (0.5, 0.5) **/
    fprintf(stdout, "\n(a) Find the nearest neighbor of (0.5, 0.5)\n\n");
    if( (start = times(&tmsstart)) == -1 )
        err_sys("time error");

    Element<double> query = {{0.5, 0.5}};
    for( int i = 0; i < FOR_NNPERSEC; ++i)
        myKdtree.NNSearch(query);
    myKdtree.printNNSearch();

    if( (end = times(&tmsend)) == -1 )
        err_sys("time error");

    /** (b) How many points there are in the rectangle (0.3, 0.3),(0.3, 0.41),(0.6, 0.3),(0.6, 0.41)? **/
    fprintf(stdout, "\n(b) How many points there are in the rectangle.\n\n");
    double range[2][2] = { {0.3, 0.6}, {0.3, 0.41} };
    FILE *fp = fopen("./output/result", "w");
    myKdtree.rangeSearch(range);
    myKdtree.printRangeSearchRes(range, fp);
    fclose(fp);

    /** (c) How many nearest neighbor calculations can your 2d-tree implementation perform per second **/
    fprintf(stdout, "\n(c) How many nearest neighbor calculations can your 2d-tree implementation perform per second.\n\n");
    static long clktck = 0;
    if( clktck == 0 )
        if( (clktck = sysconf(_SC_CLK_TCK)) < 0 )
            err_sys("sysconf error");
    double nn_persec = FOR_NNPERSEC / ((tmsend.tms_utime - tmsstart.tms_utime) / (double) clktck);
    fprintf(stdout, "    find %lf nearest neighbor per second.\n", nn_persec);

    
    return 0;
}
/**
void miniConsole() 
{
    fprintf(stdout, "what do you want to do?\n")
    string cmd

    while(1)
    {
        char
    fgets(buf, sizeof(buf), stdin) != NULL
     
    
    }
}**/
