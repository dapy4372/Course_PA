#include <stdlib.h>
#include <sys/wait.h>
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
void err_sys(const char *);
void err_quit(const char *);
void TELL_PARENT(pid_t , int);
void WAIT_PARENT(int);
void TELL_CHILD(int);
void WAIT_CHILD(int);

int main(int argc, char **argv)
{
    
    /* give FIFO name */
    char FIFO_name1[] = "BACK";
    char FIFO_name2[] = "A";

    /* make FIFOs */
    mkfifo(FIFO_name1, 0777);
    mkfifo(FIFO_name2, 0777);

    /* pipe for avoiding race condition */
    int fd[2];
    if(pipe(fd) < 0)
        err_sys("pipe error");
   
    pid_t pid;
    if( (pid = fork()) < 0 )
        err_sys("fork error");
    else if(pid == 0){
        if( ( pid = fork() ) < 0)
            err_sys("fork error");
        else if(pid > 0){
            TELL_CHILD(fd[1]);
            exit(0);
        }
        else{
            /* first child go first */
            WAIT_PARENT(fd[0]);
            close(fd[0]);
            close(fd[1]);
            if( execl("./gradchild", "gradchild", (char*)0) )  
                err_sys("execl error");
        }
    }

    if( waitpid(pid, NULL, 0) != pid)
        err_sys("waitpid error");

    int st;
    
    /* read from parent */
    char msg1[BUF_SIZE];
    char msg2[BUF_SIZE];
    char buf[BUF_SIZE];
    //fscanf(stdin, "%s %s", msg1,msg2);
    //fprintf(stdout, "%s %s \n", msg1, msg2);
    int ii[4];
    
    fscanf(stdin, "%d %d %d %d", &ii[0], &ii[1], &ii[2], &ii[3]);
//    fprintf(stdout, "%d %d %d %d\n", ii[0], ii[1], ii[2], (ii[3]+1));
//    fflush(stdout);

    int fdr = open(FIFO_name1, O_RDWR);
    FILE *fpr = fdopen(fdr,"r"); 

    fd_set reply_set, ready_read_set;
    FD_ZERO(&reply_set);
    FD_ZERO(&ready_read_set);
    int max_fd = 0;
    FD_SET(fdr, &reply_set);
    if(fdr > max_fd)
        max_fd = fdr;

    struct timeval timeout;

    /* sent a message to gradchild A */
    int fdw;
    FILE *fpw;
    fdw = open(FIFO_name2, O_RDWR);
    fpw = fdopen(fdw, "w");

    /* sent to grandchild*/
    fprintf(fpw, "%d %d %d %d\n", ii[0], ii[1], ii[2], ii[3]);
    fflush(fpw);

   
    while(1){
        ready_read_set = reply_set;
        timeout.tv_sec = 10;
        timeout.tv_usec = 0;
        int sl;
        sl = select(max_fd+1, &ready_read_set, NULL, NULL, &timeout);
        if( sl == -1 )
            perror( "select error\n");

        if(FD_ISSET(fdr, &ready_read_set)){
            int i[4];
            fscanf(fpr, "%d %d %d %d", &i[0], &i[1], &i[2], &i[3]);
            //read(fdr, buf, BUF_SIZE);
            fprintf(stdout,"%d %d %d %d\n", i[0], i[1], i[2], (i[3]+1));
            fflush(stdout);
            //fprintf(stderr,"%d %d %d %d", i[0], i[1], i[2], i[3]);
        }
    }
    exit(0);
}

void err_sys(const char * x)
{
    perror(x);
}

void err_quit(const char * x)
{
    perror(x);
    exit(1);
}

void TELL_PARENT(pid_t pid, int fd)
{
    if(write(fd, "c", 1) != 1)
        err_sys("write error");
}

void WAIT_PARENT(int fd)
{
    char c;
    if(read(fd, &c, 1) != 1)
        err_sys("read error");
    if(c != 'p')
        err_quit("WAIT_PARENT: incorrect data");
}

void TELL_CHILD(int fd)
{
    if(write(fd, "p", 1) != 1)
      err_sys("write error");
}
void WAIT_CHILD(int fd)
{
    char c;
    if(read(fd, &c, 1) != 1)
        err_sys("read error");
    if(c != 'c')
        err_quit("WAIT_CHILD: incorrect data");
}
