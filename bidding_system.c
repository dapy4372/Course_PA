#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#define BUF_SIZE 128
#define NSEC 1e9
void err_sys(const char *);
void read_testdata(const char *);
void WAIT_PARENT(int);
void TELL_CHILD(int);
void writelog(const char *, int);
void proccustomer(float);
void sighandler(int); 
void ordhandler(int);
void sig_int(int);

pid_t pid;
FILE *fpr;
int fdw;
sigset_t allmask;
sigset_t waitmask;
int customer_num[3];
int serial_num[3];

int main(int argc, char **argv)
{
    if(argc != 2){
        fprintf(stderr, "Error! Usage: $./bidding_system [test_data]\n");
        exit(EXIT_FAILURE);
    }
    char *testdata_filename = argv[1];
    read_testdata(testdata_filename);

    sigset_t orimask;
    sigemptyset(&orimask);
    sigfillset(&allmask);
    sigprocmask(SIG_BLOCK, &allmask, &orimask);

    // register signal
    struct sigaction act;
    act.sa_handler = sighandler;
    sigemptyset(&act.sa_mask);
    if(sigaction(SIGUSR1, &act, NULL) < 0 || sigaction(SIGUSR2, &act, NULL) < 0)
        exit(EXIT_FAILURE);
    sigaddset(&(act.sa_mask), SIGUSR1);
    sigaddset(&(act.sa_mask), SIGUSR2);

    // double fork customer
    int pipe_fd[2];
    pipe(pipe_fd);
    if((pid = fork()) < 0)
        err_sys("fork error\n");
    else if(pid == 0){    // child
        dup2(pipe_fd[0], STDIN_FILENO);
        dup2(pipe_fd[1], STDOUT_FILENO);
        if(execl("./customer", "customer", testdata_filename, (char*)0) == -1)
            err_sys("execl error");
    }
    else{    // parent
        if(waitpid(pid, NULL, 0) != pid)
            err_sys("waitpid error");
        close(pipe_fd[1]);
    }

    // test
    /*
    if(signal(SIGUSR1, sighandler) == SIG_ERR)
        err_sys("err");
    sigprocmask(SIG_SETMASK, &orimask, NULL);
    while(1){
        pause();
        fprintf(stderr,"##debug\n");
    }*/

    fpr = fopen(argv[1], "r");
    fdw = open("./bidding_system_log", O_WRONLY);
    
    bool done_allcustomer = false;
    fd_set fds, reply_fds;
    FD_ZERO(&fds);
    FD_ZERO(&reply_fds);
    FD_SET(pipe_fd[0], &fds);

    sigprocmask(SIG_SETMASK, &orimask, NULL);
    pause();
    fprintf(stderr,"##debug\n");
    while(!done_allcustomer){
        reply_fds = fds;
        int res;
    //    res = pselect(pipe_fd[0]+1, &reply_fds, NULL, NULL, NULL, &waitmask);
        res = select(pipe_fd[0]+1, &reply_fds, NULL, NULL, NULL);
        if(res < 0 && errno != EINTR) // interupted by signal
            err_sys("select");
        else if(res == 0)
            continue;
        if(FD_ISSET(pipe_fd[0], &reply_fds)){
            ordhandler(pipe_fd[0]);
        }
        if(memcmp(customer_num, serial_num, sizeof(customer_num)) == 0)
            done_allcustomer = true;
    }

    fclose(fpr);
    close(fdw);

    return 0;
}

void sig_int(int signo)
{
    if(signo == SIGINT)
        fprintf(stderr, "receive SIGINT\n");
}

void err_sys(const char * x)
{
    perror(x);
    exit(0);
}

void read_testdata(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    int code;
    float sendtime;
    char buf[BUF_SIZE];
    while(fgets(buf, BUF_SIZE, fp)){
        sscanf(buf, "%d %f\n", &code, &sendtime);
        fprintf(stderr, "%d %f\n", code, sendtime);
        customer_num[code] += 1; 
    }
    fclose(fp);
}

void WAIT_PARENT(int fd)
{
    char c;
    if(read(fd, &c, 1) != 1)
        err_sys("read error");
    if(c != 'p')
        err_sys("WAIT_PARENT: incorrect data");
}

void TELL_CHILD(int fd)
{
    if(write(fd, "p", 1) != 1)
      err_sys("write error");
}

void writelog(const char *status, int code)
{
    sigset_t oldmask;
    // write "receive message" to log file 
    if(sigprocmask(SIG_SETMASK, &allmask, &oldmask) < 0)
        err_sys("SIG_SETMASK error");
    char buf[BUF_SIZE];
    snprintf(buf, BUF_SIZE, "%s %d %d\n", status, code, serial_num[code]);
    //TODO debug
    //write(fdw, buf, strlen(buf)); 
    fprintf(stderr, "%s", buf);
    if(sigprocmask(SIG_SETMASK, &oldmask, NULL) < 0)
        err_sys("SIG_SETMASK error");
}

void proccustomer(float proctime)
{
    struct timespec reqtp, remtp;
    reqtp.tv_sec = (int)proctime;
    reqtp.tv_nsec = (proctime - (int)proctime) * NSEC;
    int ret;
again:
    if(ret = nanosleep(&reqtp, &remtp) < 0){
        if(errno == EINTR){
            reqtp = remtp;
            goto again;
        }
        else if(ret < 0 && errno != EINTR)
           err_sys("ordhandler sleep error");
    }
}

void sighandler(int signo)
{
    if(signo == SIGUSR1){
        writelog("receive", 1);
        proccustomer(0.5);
        kill(SIGUSR1, pid);
        writelog("finish", 1);
        serial_num[1] += 1;
    }
    else if(signo == SIGUSR2){
        writelog("receive", 2);

        sigset_t oldmask;
        if(sigprocmask(SIG_SETMASK, &allmask, &oldmask) < 0)
            err_sys("SIG_SETMASK error");
        proccustomer(0.2);
        if(sigprocmask(SIG_SETMASK, &oldmask, NULL) < 0)
            err_sys("SIG_SETMASK error");

        kill(SIGUSR2, pid);

        writelog("finish", 2);
        serial_num[2] += 1;
    }
    else
        err_sys("signo error");
}

void ordhandler(int tmpfd)
{
   writelog("receive", 0);
   char buf[BUF_SIZE];
   read(tmpfd, buf, BUF_SIZE);
   proccustomer(1.0);
   kill(SIGINT, pid);
   writelog("finish", 0);
   serial_num[0] += 1;       
}
