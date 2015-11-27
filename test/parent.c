#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/select.h>
#include <string.h>

#define READ_END 0
#define WRITE_END 1
#define BUF_SIZE 128

void err_sys(const char *);
void err_quit(const char *);
void TELL_PARENT(pid_t , int);
void WAIT_PARENT(int);
void TELL_CHILD(int);
void WAIT_CHILD(int);

int main(int argc, char **argv)
{
    //int host_num = 
    //int player_num =
    int p_to_c[2], c_to_p[2];
    pid_t pid;
    FILE *fpr;
    FILE *fpw;

    if( pipe(p_to_c) < 0 || pipe(c_to_p) < 0 )
        perror("pipe error\n");

    if( (pid = vfork()) < 0)
        perror( "fork error\n");
    else if (pid == 0) {    /* beginning of first child */
        if( ( pid = fork() ) < 0 )
            err_sys("fork error");
        else if(pid == 0){    /* beginning of second child */

            /* first child go first */
            /* child to parent pipe cuz it doesn't dup2 */
            WAIT_PARENT(c_to_p[0]);
            
            /* redirect stdin and stdout */
            dup2( p_to_c[0], STDIN_FILENO );
            dup2( c_to_p[1], STDOUT_FILENO );

            /* close unnecessary fd */
            close( p_to_c[0]);
            close( p_to_c[1]);
            close( c_to_p[0]);
            close( c_to_p[1]);
            
            /* exec */
            if( execl("./child", "child", (char*)0 ) )
                err_sys( "execl error");

        }     /* end of second child */ 
        else {
            TELL_CHILD(c_to_p[1]);
            _exit(0); 
        }    /* end of first child */
        
    }    /* end of fork child to avoid Zombie */
    else{
        //if(waitpid(pid, NULL, 0) != pid)
          //  err_sys("waitpid error");

        /* close unnecessary fd */
        close(p_to_c[0]);
        close(c_to_p[1]);

        fpw = fdopen(p_to_c[1], "w");
        fpr = fdopen(c_to_p[0], "r");
         
        int i[4]={0,1,2,3};
        //printf("%s\n", buf);
        fprintf(fpw, "%d %d %d %d\n", i[0], i[1], i[2], i[3]);
        fflush(fpw);
    }

    fd_set reply_set, ready_read_set;
    FD_ZERO(&reply_set);
    FD_ZERO(&ready_read_set);

    int max_fd = 0;
    FD_SET(c_to_p[0], &reply_set);
    if(c_to_p[0] > max_fd)
        max_fd = c_to_p[0];

    struct timeval timeout;
    while(1){
        
        //fprintf(stderr, "max_fd = %d, c_to_p[0] = %d", max_fd, c_to_p[0]);
        ready_read_set = reply_set;
        
        timeout.tv_sec = 5;
        timeout.tv_usec = 0;
        int sl;
        sl = select(max_fd+1, &ready_read_set, NULL, NULL, &timeout);
        if( sl == -1 )
            err_sys( "select error\n");

        char buf2[BUF_SIZE];
        int tmp[4];
        //printf("find should be -1\n find = %d\n", find);

        if(FD_ISSET(c_to_p[0], &ready_read_set)){
            fscanf(fpr, "%d %d %d %d", &tmp[0], &tmp[1], &tmp[2], &tmp[3]); 
            fprintf(stderr,"%d %d %d %d", tmp[0], tmp[1], tmp[2], tmp[3]);
            //fflush(stdout);
        }


    }
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
