#include <QApplication>
#include <unistd.h>
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    int pid = fork();
    if( pid == 0 ) {
        system("ssh -p 1234 pi@140.112.174.139 '/home/pi/Roy/es_finalproject/cam/run.sh 10000'");
        exit(0);
    }
    QApplication app(argc, argv);
    MainWindow mainwindow;
    return app.exec();
}

