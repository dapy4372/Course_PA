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
# define BUFSIZE 128
using namespace std;

void miniConsole(KdTree<double> &);

int main(int argc, char *argv[])
{
    if(argc != 2) {
        cout << "Usage: \n"
             << "      " << argv[0] << " <input file>\n\n"
             << "Example: \n"
             << "      " << argv[0] << " ./data/input1.txt\n\n";
        exit(0);
    }

    cout << "#########################################################################\n";
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

    cout << "#########################################################################\n";

    miniConsole(myKdtree);
    
    return 0;
}

void miniConsole(KdTree<double> &myKdtree)
{
    while(1)
    {
        cout << "\n\nWhat do you want to do?\n\n"
             << "Options:\n"
             << "     i   - insert a node\n"
             << "     d   - delete a node\n"
             << "     r   - range search\n"
             << "     n   - nearest neighbor\n"
             << "     q   - exit\n\n"
             << "> ";
        char cmd;
        cin >> cmd;
        if( cmd == 'q' )
            break;
        Element<double> el;
        switch(cmd) {
            case 'i':
                cout << "# Format:\n"
                     << "> <number> <number>\n\n"
                     << "# Example:\n"
                     << "> 0.5 0.5\n" 
                     << "> ";
                cin >> el.keys[0] >> el.keys[1];
                myKdtree.insert(el);
                cout << "# Finish Insertion!\n\n";
                break;
            case 'd':
                cout << "# Format:\n"
                     << "> <number> <number>\n\n"
                     << "# Example:\n"
                     << "> 0.5 0.5\n"
                     << "> ";
                cin >> el.keys[0] >> el.keys[1];
                myKdtree.deleteNode(el);
                cout << "# Finish deletion!\n\n";
                break;
            case 'n':
                cout << "# Format:\n"
                     << "> <number> <number>\n\n"
                     << "# Example:\n"
                     << "> 0.5 0.5\n"
                     << "> ";
                cin >> el.keys[0] >> el.keys[1];
                myKdtree.NNSearch(el);
                myKdtree.printNNSearch();
                cout << "# Finish nearest neighbor!\n\n";
                break;
            case 'r':
                double range[2][2];
                cout << "# Format:\n"
                     << "> <x-dim range1> <x-dim range2> <y-dim range1> <y-dim range2>\n\n"
                     << "# Example:\n"
                     << "> 0.3 0.4 0.3 0.6\n"
                     << "> ";
                cin >>range[0][0] >> range[0][1] >> range[1][0] >> range[1][1];
                FILE *fp = fopen("./output/result", "w");
                myKdtree.rangeSearch(range);
                myKdtree.printRangeSearchRes(range, fp);
                fclose(fp);
                cout << "# Finish range search!\n\n";
                break;
        }
    }
}
