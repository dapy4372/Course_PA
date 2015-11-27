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

#define BUF_SIZE 128
#define MAX_ROUND 10
#define PLAYER_NUM 4

int main(int argc, char **argv)
{
    //char tmp[] = "first sent from child";
    //fprintf(stdout, "%s\n", tmp);
    //fflush(stdout);
    /* give FIFO name */
    char FIFO_name1[] = "BACK";
    char FIFO_name2[] = "A";

    int fdw = open(FIFO_name1, O_RDWR);
    FILE *fpw = fdopen(fdw, "w");
    int fdr = open(FIFO_name2, O_RDWR);
    FILE *fpr = fdopen(fdr, "r");
    /* get msg from child */
    //lseek(fdr, 0, SEEK_SET);
    int tmp[4];
    fscanf(fpr, "%d %d %d %d", &tmp[0], &tmp[1], &tmp[2], &tmp[3]);
    //read(fdr, buf, BUF_SIZE);
    //fprintf(stderr, "IN G: %s\n", buf);
    //FILE *tmp = fdopen(stdout, "w");
    /* sent msg to child*/
    fprintf(fpw, "%d %d %d %d\n", tmp[0], tmp[1], tmp[2], (tmp[3]+1));
    fflush(fpw);
    //scanf(buf2, "%s\n", buf);
    //write(fdw, buf2, sizeof(buf2));
    //fsync(fdw);
    //printf("IN G: %s", buf);
    //fprintf(fpw, "%s\n", buf);
    //fflush(fpw);

    return 0;
}
