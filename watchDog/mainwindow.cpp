#include "mainwindow.h"

#include <iostream>
 
MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    main_window = new QWidget; 
    main_window->resize(1200, 800);
    layout = new QGridLayout;

    list_widget = new QListWidget;
    list_widget->setIconSize(QSize(120,120));
    list_widget->setFixedWidth(400);
    layout->addWidget(list_widget, 0, 0);

    stacked_layout = new QStackedLayout;
    layout->addLayout(stacked_layout, 0, 1);

    update_button = new QPushButton("Update");
    quit_button = new QPushButton("Quit");
    play_button = new QPushButton("Play");
    layout->addWidget(update_button, 1, 0);
    layout->addWidget(quit_button, 2, 0);
    layout->addWidget(play_button, 1, 1);
    connect(update_button, SIGNAL(clicked()), this, SLOT(on_addButton_clicked()));
    connect(quit_button, SIGNAL(clicked()), main_window, SLOT(close()) );

    QObject::connect(list_widget, SIGNAL(currentRowChanged(int)), stacked_layout, SLOT(setCurrentIndex(int)));
    QObject::connect(list_widget, SIGNAL(currentRowChanged(int)), this, SLOT(playVideo(int)));
    main_window->setLayout(layout);
    main_window->show();
}
 
MainWindow::~MainWindow() {}

QVector< QPair<QString, int> > readTable(const QString &filename)
{
    QFile inputFile(filename);
    QVector< QPair<QString, int> > img_vec;
    if(inputFile.open(QIODevice::ReadOnly)) {
        QTextStream in(&inputFile);
        while(!in.atEnd()) {
            QString line = in.readLine();
            QString img_path = "./data/image/" + line.split(" ").at(0);
            int img_time = line.split(" ").at(1).toInt();
            QPair<QString, int> a(img_path, img_time);

            img_vec << a;
        }
        inputFile.close();
    }
    return img_vec;
}

void MainWindow::on_addButton_clicked()
{
    QDir dir("./data/table");
    QFileInfoList list = dir.entryInfoList(QDir::Files);
    Q_FOREACH(QFileInfo finfo, list) {
        QString table_path = finfo.filePath();
        QString video_path = table_path;
        video_path.replace("table", "video").replace("txt", "mp4");
        QVector< QPair<QString, int> > img_vec = readTable(table_path);
        for( int i = 0; i < img_vec.size(); ++i ) {
            Phonon::VideoPlayer *player = new Phonon::VideoPlayer(Phonon::VideoCategory, main_window);
            player->load(video_path);
            player->play();
            player->pause();
            player->seek(img_vec.at(i).second);
            stacked_layout -> addWidget(player);
            QListWidgetItem *item = new QListWidgetItem(QIcon(img_vec.at(i).first), "show");
            MyItem *myitem = new MyItem(list_widget -> count(), item, player, video_path, img_vec.at(i).second);
            list_widget -> insertItem(list_widget -> count(), item);
            myitem_vec.append(myitem);
        }
    }
}

void MainWindow::playVideo(int idx)
{
    MyItem *cur_item = myitem_vec.at(idx);
    //cur_item->player->load(cur_item->video_path);
    //while(!cur_item->player->mediaObject()->isSeekable()) {
    //    qDebug() << "123";
    //}
    cur_item->player->seek(cur_item->img_time);
    qDebug() << cur_item->img_time;
    cur_item->player->play();
}
