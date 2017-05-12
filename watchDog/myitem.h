#ifndef MYITEM_H
#define MYITEM_H
 
#include <QWidget>
#include <QListWidget>
#include <QString>
#include <phonon>
#include <phonon/VideoPlayer>
#include <phonon/VideoWidget>
#include <phonon/MediaObject>
#include <phonon/MediaSource>
/* My Includes */
 
class MyItem : public QWidget
{
    Q_OBJECT
public:
    explicit MyItem(QWidget *parent = 0);
    MyItem(int id, QListWidgetItem *it, Phonon::VideoPlayer *p, QString s, int t) : _id(id), item(it), player(p), video_path(s), img_time(t), _seekable(false) { }
    ~MyItem() { }
    int getID();
    int _id;
    QListWidgetItem *item;
    Phonon::VideoPlayer *player;
    QString video_path;
    int img_time;
    bool _seekable;
 
public slots:
    void playVideo();
    void testSeek(bool);
 
private:
};
 
#endif 
