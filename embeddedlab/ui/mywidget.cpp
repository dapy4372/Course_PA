#include <QPushButton>
#include "mywidget.h"
#include "mainwindow.h"

MyWidget::MyWidget(QWidget *parent) : QWidget(parent)
{
    setFixedSize(1000, 1000);
    QVector<QPushButton *> btn_ptr;
    int num_button = 20;
    int num_perline = 5;
    int x = 0, y = 0;
    for(int i = 0; i < num_button; i++){
        QPushButton *b = new QPushButton("Hellow" + QString::number(i), this);
        b -> setIcon(QIcon("./img/" + QString::number(i) + ".png"));
        b -> setIconSize(QSize(50,50));
        b -> move(x* 150 + 30, y * 150 + 30);
        connect(b, SIGNAL(clicked()), this, SLOT(MainWindow::playVideo()));
        b -> show();
        btn_ptr << b;
        if(i != 0 && i % num_perline == 0){
            x = 0;
            y++;
        }
        else
            x++;
    }
}

void MyWidget::playVideo()
{
}
