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
    char FIFO_name1[] = "host_parent";
    char FIFO_name2[] = "host";

    /* make FIFOs */
    int idx = 0;
    mkfifo(FIFO_name1, 0666);
    mkfifo(FIFO_name2, 0666);
    
    /* get player id from bidding system */
    char buf[BUF_SIZE];
    fgets(buf, BUF_SIZE, stdin);
    fprintf(stdout, "%s\n", buf);
    fflush(stdout);

    char msg[] = "sent from child";

    fprintf(stdout, "%s\n", msg);
    fflush(stdout);

    return 0;
}
