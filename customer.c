#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>

#define BUF_SIZE 128
void err_sys(const char *);
int main(int argc, char **argv)
{
    if(argc != 2){
        fprintf(stderr, "Error! Usage: $./costumer [test_data]\n");
        exit(EXIT_FAILURE);
    }
    
    FILE *fpr;
    FILE *fpw;
    fpr = fopen(argv[1], "r");
    fpw = fopen("./customer_log", "w");
    
    char buf[BUF_SIZE];
    int tmp, code=1;
    float sendtime, prev_sendtime = 0;
    struct timespec tim;
    while(fgets(buf, BUF_SIZE, fpr)){
        sscanf(buf, "%d %f\n", &code, &sendtime);
        fprintf(stderr, "%d %f\n", code, sendtime);
        
        int time_diff = sendtime - prev_sendtime;
        tim.tv_sec = time_diff;
        tim.tv_nsec = ((sendtime - prev_sendtime) - time_diff) * 1e9;

        if(nanosleep(&tim, NULL) < 0)
          fprintf(stderr, "no sleep");
        
        prev_sendtime = sendtime;
    }

    fclose(fpr);
    fclose(fpw);

    return 0;
}

void err_sys(const char * x)
{
    perror(x);
    exit(0);
}

