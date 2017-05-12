#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    if(argc != 2){
        fprintf(stderr, "Error! Usage: $./bidding_system [test_data]\n");
        exit(EXIT_FAILURE);
    }

    return 0;
}
