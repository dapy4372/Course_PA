#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>

#define BUF_SIZE 128
#define NSEC 1e9

void err_sys(const char *);
void sendcustomer(int code);
void sighandler(int);
FILE *fpw;


int main(int argc, char **argv)
{
    if(argc != 2){
        fprintf(stderr, "Error! Usage: $./costumer [test_data]\n");
        exit(EXIT_FAILURE);
    }
    
    struct sigaction act;
    //act.sa_handler = sighandler(int signo);


    FILE *fpr;
    fpr = fopen(argv[1], "r");
    fpw = fopen("./customer_log", "w");
    
    char buf[BUF_SIZE];
    int tmp, code;
    float sendtime, prev_sendtime = 0;
    struct timespec tim;
    while(fgets(buf, BUF_SIZE, fpr)){
        sscanf(buf, "%d %f\n", &code, &sendtime);
        fprintf(stderr, "%d %f\n", code, sendtime);
        
        float time_diff = sendtime - prev_sendtime;
        tim.tv_sec = (int)time_diff;
        tim.tv_nsec = (time_diff - (int)time_diff) * NSEC;
        fprintf(stderr, " == sec = %d; nsec=%ld == \n", tim.tv_sec, tim.tv_nsec);

        // TODO avoid signal to wake sleep
        if(nanosleep(&tim, NULL) < 0)
            fprintf(stderr, "no sleep");

        sendcustomer(code);
        
        

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

void sendcustomer(int code)
{
    switch(code){
        case 0:{
            char *str = "ordinary\n";
            write(STDOUT_FILENO, str, strlen(str));
            break;
            }
        case 1:
            kill(getppid(), SIGUSR1);
            break;
        case 2:
            kill(getppid(), SIGUSR2);
            break;
    }
}

void sighandler(int signo)
{
    if(signo == SIGINT){
       // fprintf(fpw, "finish %d %d", )
    
    }


}
