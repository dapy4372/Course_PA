#include "myitem.h"

int MyItem::getID()
{
    return _id;
}

void MyItem::playVideo()
{
    player->play();
}
