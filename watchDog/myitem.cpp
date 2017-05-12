#include "myitem.h"

int MyItem::getID()
{
    return _id;
}

void MyItem::playVideo()
{
    player->play();
}

void MyItem::testSeek(bool b)
{
    //qDebug() << "it's seekable!!!!" << endl << "#################################################";
    qDebug() << endl << "#################################################";
    qDebug() << b <<endl;
    _seekable = b;
}
