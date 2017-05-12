#include "mainwindow.h"
#include <unistd.h>

#include <iostream>
 
QMap<QString, int> read_grp_list();
QVector< QPair<QString, QPair<int, int> > > readTable(const QString &);

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    main_window = new QWidget; 
    main_window -> resize(1200, 800);
    layout = new QGridLayout;

    list_widget = new QListWidget;
    list_widget -> setIconSize(QSize(120,120));
    list_widget -> setFixedWidth(400);
    layout -> addWidget(list_widget, 0, 0);

    update_button = new QPushButton("Update");
    quit_button = new QPushButton("Quit");
    layout -> addWidget(update_button, 1, 0);
    layout -> addWidget(quit_button, 2, 0);
    connect(update_button, SIGNAL(clicked()), this, SLOT(on_addButton_clicked()));
    connect(quit_button, SIGNAL(clicked()), main_window, SLOT(close()) );

    //QObject::connect(list_widget, SIGNAL(itemClicked(QListWidgetItem *)), this, SLOT(handleVideo(QListWidgetItem *)));
    //QObject::connect(list_widget, SIGNAL(currentItemChanged(QListWidgetItem *, QListWidgetItem *)), this, SLOT(handleVideo(QListWidgetItem *, QListWidgetItem *)));
    QObject::connect(list_widget, SIGNAL(itemClicked(QListWidgetItem *)), this, SLOT(openNewWindow(QListWidgetItem *)));
    QObject::connect(list_widget, SIGNAL(currentItemChanged(QListWidgetItem *, QListWidgetItem *)), this, SLOT(openNewWindow(QListWidgetItem *, QListWidgetItem *)));
    main_window -> setLayout(layout);
    main_window -> show();
}
 
MainWindow::~MainWindow() {}

QVector< QPair<QString, QPair<int, int> > > readTable(const QString &filename)
{
    QMap<QString, int> grp_list = read_grp_list();
    QFile inputFile(filename);
    QVector< QPair<QString, QPair<int, int> > > img_vec;
    if(inputFile.open(QIODevice::ReadOnly)) {
        QTextStream in(&inputFile);
        while(!in.atEnd()) {
            QString line = in.readLine();
            QString img_filename = line.split(" ").at(0);
            QString img_path = "./data/image/" + img_filename;

            QMap<QString, int>::const_iterator iter = grp_list.constFind(img_filename);
            if( iter != grp_list.constEnd() ) {
                int img_time = line.split(" ").at(1).toInt();
                QPair<int, int> img_info(img_time, iter.value());
                QPair<QString, QPair<int, int> > a(img_path, img_info);
                img_vec << a;
            }
        }
        inputFile.close();
    }
    return img_vec;
}

QMap<QString, int> read_grp_list()
{
    QMap<QString, int> grp_list;
    QFile inputFile("./data/grp_list");
    if(inputFile.open(QIODevice::ReadOnly)) {
        QTextStream in(&inputFile);
        while(!in.atEnd()) {
            QStringList line = in.readLine().split(" ");
            grp_list[line.at(1)] = line.at(0).toInt();
        }
    }
    return grp_list;
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
            QVector< QPair<QString, QPair<int, int> > > img_vec = readTable(table_path);
            Phonon::VideoPlayer *player = new Phonon::VideoPlayer(Phonon::VideoCategory, main_window);
            player -> load(video_path);
            qDebug() << video_path;

            for( int i = 0; i < img_vec.size(); ++i ) {
                qDebug() << img_vec.at(i);
                int curr_groupidx = (img_vec.at(i).second).second;
                QListWidgetItem *item = new QListWidgetItem(QIcon(img_vec.at(i).first), img_vec.at(i).first);

                QMap<int, NewWindow *>::const_iterator iter = newWindow_map.constFind(curr_groupidx); // second for grouping idx
                if( iter != newWindow_map.constEnd() ) {
                    MyItem *myitem = new MyItem(list_widget -> count(), item, player, video_path, (img_vec.at(i).second).first);
                    iter.value() -> addItem(myitem);
                    qDebug() << iter.value() -> myitem_vec; 
                    qDebug() << "old nw\n\n";
                }
                else {// no grouping id in the map now, creat a newWindow
                    MyItem *myitem = new MyItem(list_widget -> count(), item, player, video_path, (img_vec.at(i).second).first);
                    NewWindow *nw = new NewWindow(); // use map size as newwindow id
                    nw -> addItem(myitem);
                    QListWidgetItem *nw_item = new QListWidgetItem(QIcon(img_vec.at(i).first), img_vec.at(i).first);
                    nw_item -> setWhatsThis(QString::number(curr_groupidx));
                    qDebug() << "new nw  " << curr_groupidx;
                    list_widget -> insertItem(list_widget -> count(), nw_item);
                    newWindow_map[curr_groupidx] = nw; 
                }
            }
        }
    }
}

void MainWindow::openNewWindow(QListWidgetItem *curr)
{
    NewWindow *curr_newWindow = newWindow_map[curr -> whatsThis().toInt()];
    curr_newWindow -> main_window -> show();
}

void MainWindow::openNewWindow(QListWidgetItem *curr, QListWidgetItem *prev)
{
    //qDebug() << curr->whatsThis().toInt();

    // check if it is the first time to call this function
    // if true, the prev will be null
    if(prev) {
        qDebug() << prev -> whatsThis().toInt();

        NewWindow *prev_item = newWindow_map[prev -> whatsThis().toInt()];
        prev_item -> main_window -> hide();
    }
}
