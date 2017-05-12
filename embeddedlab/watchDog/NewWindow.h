#ifndef NEWWINDOW_H
#define NEWWINDOW_H
 
#include <QMainWindow>
#include <QWidget>
#include <QListWidget>
#include <QString>
#include <QGridLayout> 
#include <QListWidget>
#include <QPushButton>
#include <QDebug>
#include <phonon>
#include <phonon/VideoPlayer>
#include <phonon/VideoWidget>
#include <phonon/MediaObject>
#include <phonon/MediaSource>
/* My Includes */
#include "myitem.h"
 
class NewWindow : public QWidget
{
    Q_OBJECT
public:
    explicit NewWindow(QWidget *parent = 0);
    //NewWindow(int id, MyItem *it, QWidget *parent = 0);
    //explicit NewWindow(int id, MyItem *it, QWidget *parent = 0);
    ~NewWindow() { }
    int _id;
    QVector<MyItem *> myitem_vec;
    int curr_myitem_idx;

    QWidget *main_window;
    QGridLayout *layout;
    QListWidget *list_widget;
    QPushButton *update_button;
    QPushButton *quit_button;

    void addItem(MyItem *);
 
public slots:
    void handleVideo(QListWidgetItem *);
    void handleVideo(QListWidgetItem *, QListWidgetItem *);
    void on_addButton_clicked();
 
};
 
#endif 
