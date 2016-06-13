#ifndef MAINWINDOW_H
#define MAINWINDOW_H
 
#include <QMainWindow>
#include <QGridLayout> 
#include <QStackedLayout>
#include <QListWidget>
#include <QPushButton>
#include <QLabel>
#include <QTextEdit>
#include <QString>
#include <QDebug>
#include <QFile>
#include <QTextStream>
#include <QDir>
#include <phonon>
#include <phonon/VideoPlayer>
#include <phonon/VideoWidget>
#include <phonon/MediaObject>
#include <phonon/MediaSource>
/* My Includes */
#include "myitem.h"
 
class MainWindow : public QMainWindow
{
    Q_OBJECT
 
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    QWidget *main_window;
    QGridLayout *layout;
    QListWidget *list_widget;
    QStackedLayout *stacked_layout;
    QPushButton *update_button;
    QPushButton *quit_button;
    QPushButton *play_button;
    Phonon::VideoPlayer *player;
    QVector<MyItem *> myitem_vec;
 
private slots:
    void on_addButton_clicked();
    void playVideo(int);
 
private:
    QWidget *_parent;
};
 
#endif // MAINWINDOW_H
