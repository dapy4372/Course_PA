#include "mainwindow.h"
#include <unistd.h>

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

    //stacked_layout = new QStackedLayout;
    //layout->addLayout(stacked_layout, 0, 1);

    update_button = new QPushButton("Update");
    quit_button = new QPushButton("Quit");
    play_button = new QPushButton("Play");
    layout->addWidget(update_button, 1, 0);
    layout->addWidget(quit_button, 2, 0);
    layout->addWidget(play_button, 1, 1);
    connect(update_button, SIGNAL(clicked()), this, SLOT(on_addButton_clicked()));
    connect(quit_button, SIGNAL(clicked()), main_window, SLOT(close()) );

    //QObject::connect(list_widget, SIGNAL(currentRowChanged(int)), stacked_layout, SLOT(setCurrentIndex(int)));
    //QObject::connect(list_widget, SIGNAL(currentRowChanged(int)), this, SLOT(playVideo(int)));
    QObject::connect(list_widget, SIGNAL(itemClicked(QListWidgetItem *)), this, SLOT(handleVideo(QListWidgetItem *)));
    QObject::connect(list_widget, SIGNAL(currentItemChanged(QListWidgetItem *, QListWidgetItem *)), this, SLOT(handleVideo(QListWidgetItem *, QListWidgetItem *)));
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
        QMap<QString, QString>::const_iterator iter = _path_map.constFind(video_path);
        if( iter == _path_map.constEnd() ) {
            _path_map[video_path] = table_path;
            video_path.replace("table", "video").replace("txt", "mp4");
            QVector< QPair<QString, int> > img_vec = readTable(table_path);
            Phonon::VideoPlayer *player = new Phonon::VideoPlayer(Phonon::VideoCategory, main_window);
            player -> load(video_path);
            player -> play();
            player -> pause();
            //QObject::connect(player -> mediaObject(), SIGNAL(prefinishMarkReached(qint32)), player -> mediaObject(), SLOT(seek(qint32)));
            for( int i = 0; i < img_vec.size(); ++i ) {
                //player->mediaObject()->setCurrentSource(video_path);
                //stacked_layout -> addWidget(player);
                layout -> addWidget(player, 0, 1);
                player->hide();
                QListWidgetItem *item = new QListWidgetItem(QIcon(img_vec.at(i).first), img_vec.at(i).first);
                item->setWhatsThis(QString::number(list_widget->count()));

                MyItem *myitem = new MyItem(list_widget -> count(), item, player, video_path, img_vec.at(i).second);
                QObject::connect(player -> mediaObject(), SIGNAL(hasVideoChanged(bool)), myitem, SLOT(testSeek(bool)));


                list_widget -> insertItem(list_widget -> count(), item);
                myitem_vec.append(myitem);
            }
        }
    }
}

void MainWindow::handleVideo(QListWidgetItem *curr)
{
    MyItem *curr_item = myitem_vec.at(curr -> whatsThis().toInt());
    qDebug() << "curr:" << curr_item -> img_time << endl;
    qDebug() << "State:" << curr_item -> player -> mediaObject() -> state();
    
    if( curr_item -> player -> isHidden() )
        curr_item -> player -> show();

    if( curr_item -> player -> mediaObject() -> state() == Phonon::LoadingState ) {
        qDebug() << "loading!!!";
        return;
    }

    if( curr_item -> _seekable ) {
        curr_item -> player -> seek(curr_item -> img_time);
        qDebug() << "curr:" << curr_item -> img_time << endl;
        curr_item -> player -> play();
    }
    else {
        curr_item -> player -> pause();
        qDebug() << "handle: wait for being seekable" << endl;
    }
}

void MainWindow::handleVideo(QListWidgetItem *curr, QListWidgetItem *prev)
{
    qDebug() << curr->whatsThis().toInt();

    // check if it is the first time to call this function
    // if true, the prev will be null
    if(prev) {
        qDebug() << prev->whatsThis().toInt();

        MyItem *prev_item = myitem_vec.at(prev->whatsThis().toInt());
        MyItem *curr_item = myitem_vec.at(curr->whatsThis().toInt());

        //prev_item->player->seek(prev_item->img_time);
        qDebug() << "prev:" << prev_item->img_time << endl;
        if( prev_item -> player != curr_item -> player ) {
            prev_item->player->pause();
            prev_item -> player -> hide();
            curr_item -> player -> show();
        }
    }
}
