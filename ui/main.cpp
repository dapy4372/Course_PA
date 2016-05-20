#include <QApplication>
#include <QPushButton> 
#include "mainwindow.h"
#include "mywidget.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    //MyWidget widget;
    //widget.show();
    MainWindow mainwindow;
    return app.exec();
}
