#include "mainwindow.h"
#include <QGridLayout>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    Main = new QWidget;
    //addFile = new QPushButton("Add file");
    pause = new QPushButton("pause");
    quit = new QPushButton("quit");
    play = new QPushButton("play");

    int num_img = 3;
    player = new Phonon::VideoPlayer(Phonon::VideoCategory, Main);
    //QVBoxLayout *layout = new QVBoxLayout;
    QGridLayout *layout = new QGridLayout;
    layout->setColumnMinimumWidth(0, 1200);
    layout->setColumnMinimumWidth(1, 200);
    //layout->addWidget(addFile);
    //layout->addWidget(play, 1, 0);
    layout->addWidget(pause, num_img, 1);
    layout->addWidget(quit, num_img+1, 1);
    layout->addWidget(player, 0, 0, num_img+2, 1);
    Main->setWindowTitle("Monitor");
    Main->setLayout(layout);
    Main->setGeometry(0, 0, 500, 500);
    Main->show();

    _file = "./video/fruit.mp4";
    player->load(_file);
    //connect(addFile, SIGNAL(clicked()), this, SLOT(startVideo()));
    //connect(play, SIGNAL(clicked()), this, SLOT(playVideo(int)) );
    connect(pause, SIGNAL(clicked()), this, SLOT(pauseVideo()));
    connect(quit, SIGNAL(clicked()), Main, SLOT(close()) );
    //connect(player->mediaObject(), SIGNAL(seekableChangedbool), this, SLOT(seekAnimation(bool)) );

    //setFixedSize(1000, 1000);
    QSignalMapper *signalMapper = new QSignalMapper(this);
    connect(signalMapper, SIGNAL(mapped(int)), this, SLOT(playVideo(int)));
    int startTime[3] = {6000, 16000, 27000};
    for(int i = 0; i < num_img; i++){
        QPushButton *b = new QPushButton("Person " + QString::number(i+1), this);
        b -> setIcon(QIcon("./img/" + QString::number(i) + ".png"));
        b -> setIconSize(QSize(120, 120));
        layout->addWidget(b, i, 1);
        QFont font = b->font();
        font.setPointSize(16);
        b->setFont(font);
        //b -> setIcon(QIcon("./img/abc.png"));
        signalMapper->setMapping(b, startTime[i]);
        connect(b, SIGNAL(clicked()), signalMapper, SLOT(map()));
        b -> show();
        _bptrVec << b;
    }
}

MainWindow::~MainWindow()
{

}

//void MainWindow::seekAnimation(bool seekable)
//{
//    //_seekable = seekable;
//    qDebug() << seekable;
//    player->seek(3000);
//    _seekable = seekable;
//}

//void MainWindow::startVideo()
//{
//    //_file = QFileDialog::getOpenFileName(this, tr("./video/fruit.mp4"));
//    //_file = "./video/done.mp4";
//    //player->play(_file);
//}

void MainWindow::playVideo(int location)
{
    player->seek(location);
    player->play();
}

void MainWindow::pauseVideo()
{
    player->pause();
}

void MainWindow::startTimeSet(float s)
{
    _startTime = s;
}
