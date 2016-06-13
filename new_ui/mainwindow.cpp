#include "mainwindow.h"

#include <iostream>
//#define foreach Q_FOREACH
 
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
    //stacked_layout->addWidget(new QLabel("<h1><font color=blue>caterpillar</font></h1>"));
    //stacked_layout->addWidget(new QPushButton("momor"));
    //stacked_layout->addWidget(new QTextEdit);
    //QObject::connect(list_widget, SIGNAL(currentRowChanged(int)), stacked_layout, SLOT(setCurrentIndex(int)));

    update_button = new QPushButton("Update");
    quit_button = new QPushButton("Quit");
    play_button = new QPushButton("Play");
    layout->addWidget(update_button, 1, 0);
    layout->addWidget(quit_button, 2, 0);
    layout->addWidget(play_button, 1, 1);
    connect(update_button, SIGNAL(clicked()), this, SLOT(on_addButton_clicked()));
    connect(quit_button, SIGNAL(clicked()), main_window, SLOT(close()) );
    connect(play_button, SIGNAL(clicked()), this, SLOT(playVideo()));

    QString filename = "./video/fruit.mp4";
    
    //player = new Phonon::VideoPlayer(Phonon::VideoCategory, main_window);
    //layout->addWidget(player);
    //player->load(Phonon::MediaSource ("./video/output.mp4"));
    //player->play();
    QObject::connect(list_widget, SIGNAL(currentRowChanged(int)), stacked_layout, SLOT(setCurrentIndex(int)));
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
            QString img_path = "../data/image/" + line.split(" ").at(0);
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
    QDir dir("../data/table");
    QFileInfoList list = dir.entryInfoList(QDir::Files);
    Q_FOREACH(QFileInfo finfo, list) {
        QString tmp = finfo.filePath();
        QVector< QPair<QString, int> > img_vec = readTable(tmp);
        for( int i = 0; i < img_vec.size(); ++i ) {
            QListWidgetItem *item = new QListWidgetItem(QIcon(img_vec.at(i).first), "show");
            list_widget->insertItem(list_widget->count(), item);
        }

    }
    //QDynamicButton *button = new QDynamicButton(this); 
    //button->setText("new button" + QString::number(button->getID()));
    //layout->addWidget(button);
    //button->show();
    //connect(button, SIGNAL(clicked()), this, SLOT(slotGetNumber()));
    //-----------------
    //QListWidgetItem *item = new QListWidgetItem(QIcon("./img/0.png"), "show");
    //list_widget->insertItem(list_widget->count(), item);
    //player = new Phonon::VideoPlayer(Phonon::VideoCategory, main_window);
    //stacked_layout->addWidget(player);
    //player->load(Phonon::MediaSource ("./video/output.mp4"));
    //player->play();
}

void MainWindow::playVideo()
{
    player->play();
}
 
/* Метод для удаления динамической кнопки по её номеру
 * */
void MainWindow::on_deleteButton_clicked()
{
    //for(int i = 0; i < layout->count(); i++){
    //for(int i = 0; i < splitter->count(); i++){
        //QDynamicButton *button = qobject_cast<QDynamicButton*>(splitter->itemAt(i)->widget());
        //QDynamicButton *button = qobject_cast<QDynamicButton*>(layout->itemAt(i)->widget());
      //  if(button->getID() == parent->lineEdit->text().toInt()){
         //   button->hide();
          //  delete button;
        //}
    //}
}
 
void MainWindow::slotGetNumber()
{
    //QDynamicButton *button = (QDynamicButton*) sender();
    //parent->lineEdit->setText(QString::number(button->getID()));
}
