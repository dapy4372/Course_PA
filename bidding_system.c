#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#define BUF_SIZE 128

void err_sys(const char *);
void WAIT_PARENT(int);
void TELL_CHILD(int);
  
int main(int argc, char **argv)
{
    if(argc != 2){
        fprintf(stderr, "Error! Usage: $./bidding_system [test_data]\n");
        exit(EXIT_FAILURE);
    }
    char *testdata_filename = argv[1];

    // double fork customer
    int tmp_pipe[2];
    pipe(tmp_pipe);
    pid_t pid;
    if((pid = fork()) < 0)
        err_sys("fork error\n");
    else if(pid == 0){
        if((pid = fork()) < 0)
            err_sys("fork error");
        else if(pid == 0){    /* grandchild */
            WAIT_PARENT(tmp_pipe[0]);
            dup2(tmp_pipe[0], STDIN_FILENO);
            dup2(tmp_pipe[1], STDOUT_FILENO);
            if(execl("./customer", "customer", testdata_filename, (char*)0) == -1)
                err_sys("execl error");
        }
        else{    /* child */
            TELL_CHILD(tmp_pipe[1]);
            exit(0);
        }
    }
    else{    /* parent */
        if(waitpid(pid, NULL, 0) != pid)
            err_sys("waitpid error");
        close(tmp_pipe[1]);
    }

    FILE *fpr;
    FILE *fpw;
    fpr = fopen(argv[1], "r");
    fpw = fopen("./bidding_system_log", "w");
    fclose(fpr);
    fclose(fpw);

    return 0;
}

void err_sys(const char * x)
{
    perror(x);
    exit(0);
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

