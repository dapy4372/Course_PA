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

    if( (pid = fork()) < 0){
        perror( "fork error\n");
    }
    else if (pid == 0){  //child
        dup2( p_to_c[0], STDIN_FILENO );
        dup2( c_to_p[1], STDOUT_FILENO );

        close( p_to_c[0]);
        close( p_to_c[1]);
        close( c_to_p[0]);
        close( c_to_p[1]);
        
        char msg[] = "I'm a child";
        if( execl("./child", "child", msg, (char*)0 ) )
            perror( "execl error\n");
    }
    else{  // parent
    
        close(p_to_c[0]);
        close(c_to_p[1]);

        fpw = fdopen(p_to_c[1], "w");
        fpr = fdopen(c_to_p[0], "r");

        char buf[20];
        fprintf(fpw, "12345");
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
