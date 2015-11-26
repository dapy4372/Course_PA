#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/select.h>
#include <fcntl.h>
#include <signal.h>
#include <stdbool.h>

#define BUF_SIZE 20
#define MAX_ROUND 10
#define PLAYER_NUM 4

int main(int argc, char **argv)
{
    //char tmp[] = "first sent from child";
    //fprintf(stdout, "%s\n", tmp);
    //fflush(stdout);

    /* give FIFO name */
    char FIFO_name1[] = "c_to_gradc";
    char FIFO_name2[] = "A";
    char FIFO_name3[] = "B";

    /* make FIFOs */
    int i;
    for(i = 0; i < 3; ++i){
        mkfifo(FIFO_name1, 0666);
        mkfifo(FIFO_name2, 0666);
        mkfifo(FIFO_name3, 0666);
    }
    
    /* get player id from bidding system */
    char buf[BUF_SIZE];
    fgets(buf, BUF_SIZE, stdin);
    //fprintf(stdout, "%s\n", buf);
    //fflush(stdout);

    //char msg[] = "sent from child";
    pid_t ppid = getppid();
    fprintf(stdout, "%d\n", ppid);
    fflush(stdout);

    return 0;
}
