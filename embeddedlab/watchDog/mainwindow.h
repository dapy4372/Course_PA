#ifndef MAINWINDOW_H
#define MAINWINDOW_H
 
#include <QMainWindow>
#include <QGridLayout> 
#include <QListWidget>
#include <QPushButton>
#include <QLabel>
#include <QTextEdit>
#include <QString>
#include <QDebug>
#include <QFile>
#include <QTextStream>
#include <QDir>
#include <QMap>
#include <phonon>
#include <phonon/VideoPlayer>
#include <phonon/VideoWidget>
#include <phonon/MediaObject>
#include <phonon/MediaSource>
/* My Includes */
#include "myitem.h"
#include "NewWindow.h"
 
class MainWindow : public QMainWindow
{
    Q_OBJECT
 
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    QWidget *main_window;
    QGridLayout *layout;
    QListWidget *list_widget;
    QPushButton *update_button;
    QPushButton *quit_button;
    Phonon::VideoPlayer *player;
    QVector<MyItem *> myitem_vec;
    QMap<int, NewWindow *> newWindow_map;
 
private slots:
    void on_addButton_clicked();
    //void handleVideo(QListWidgetItem *);
    //void handleVideo(QListWidgetItem *a, QListWidgetItem * b);
    void openNewWindow(QListWidgetItem *);
    void openNewWindow(QListWidgetItem *, QListWidgetItem *);
 
private:
    QMap<QString, QString> _path_map;
    QWidget *_parent;
};
 
#endif // MAINWINDOW_H
