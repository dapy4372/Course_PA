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
    
    int p_to_c[2], c_to_p[2];
    int child_pid;
    FILE *fpr;
    FILE *fpw;
    pid_t pid;

    fd_set reply_set, ready_read_set;
    FD_ZERO(&reply_set);

    if( pipe(p_to_c) < 0 || pipe(c_to_p) < 0 )
        perror("pipe error\n");
    
    int max_fd = 0;

    FD_SET(c_to_p[0], &reply_set);
    if(c_to_p[0] > max_fd)
        max_fd = c_to_p[0];

    if( (pid = fork()) < 0)
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
                err_sys( "execl error\n");

        }     /* end of second child */ 
        else {
            TELL_CHILD(c_to_p[1]);
            exit(0); 
        }    /* end of first child */
        
    }    /* end of fork child to avoid Zombie */
    else{
        if(waitpid(pid, NULL, 0) != pid)
            err_sys("waitpid error");

        /* close unnecessary fd */
        close(p_to_c[0]);
        close(c_to_p[1]);

        fpw = fdopen(p_to_c[1], "w");
        fpr = fdopen(c_to_p[0], "r");

        char buf[] = "12345";
        fprintf(fpw, "%s\n", buf);
        fflush(fpw);
    }

    while(1){
        
        //fprintf(stderr, "max_fd = %d, c_to_p[0] = %d", max_fd, c_to_p[0]);
        ready_read_set = reply_set;
        int sl;
        sl = select(max_fd+1, &ready_read_set, NULL, NULL, NULL);
        if( sl == -1 )
            perror( "select error\n");

        char buf[20];
        int cur_ready_host;
        int find = -1;
        for(cur_ready_host = 0; cur_ready_host < 1 && !find; ++cur_ready_host)
            if(FD_ISSET(c_to_p[0], &ready_read_set))
                find = 1;

        if(find){
            fgets(buf, 20, fpr); 
            printf("%s", buf);
            break;
        }
        else
            printf("not yet found");
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
