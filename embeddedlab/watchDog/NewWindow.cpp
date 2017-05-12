#include "NewWindow.h"

 
//explicit NewWindow::NewWindow(int id, MyItem *it, QWidget *parent = 0) : _id(id), item_vec(it_v), curr_myitem_idx(0)
NewWindow::NewWindow(QWidget *parent)
{ 
    curr_myitem_idx = 0;
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

    QObject::connect(list_widget, SIGNAL(itemClicked(QListWidgetItem *)), this, SLOT(handleVideo(QListWidgetItem *)));
    QObject::connect(list_widget, SIGNAL(currentItemChanged(QListWidgetItem *, QListWidgetItem *)), this, SLOT(handleVideo(QListWidgetItem *, QListWidgetItem *)));

    main_window -> setLayout(layout);
    //main_window -> show();
}

void NewWindow::addItem(MyItem *it)
{
    myitem_vec << it;
}

void NewWindow::handleVideo(QListWidgetItem *curr)
{
    qDebug () << "handle----------------------------------";
    MyItem *curr_item = myitem_vec.at(curr -> whatsThis().toInt());
    qDebug() << "curr:" << curr_item -> img_time << endl;
    qDebug() << "State:" << curr_item -> player -> mediaObject() -> state();
    
    // need to play first for being seekable
    if( curr_item -> player -> mediaObject() -> state() == Phonon::StoppedState ) {
        curr_item -> player -> play();
    }

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
        //curr_item -> player -> pause();
        qDebug() << "handle: wait for being seekable" << endl;
    }
}

void NewWindow::handleVideo(QListWidgetItem *curr, QListWidgetItem *prev)
{
    qDebug() << curr->whatsThis().toInt();

    // check if it is the first time to call this function
    // if true, the prev will be null
    if(prev) {
        qDebug() << prev -> whatsThis().toInt();

        MyItem *prev_item = myitem_vec.at(prev -> whatsThis().toInt());
        MyItem *curr_item = myitem_vec.at(curr -> whatsThis().toInt());

        //prev_item->player->seek(prev_item->img_time);
        qDebug() << "prev:" << prev_item -> img_time << endl;
        if( prev_item -> player != curr_item -> player ) {
            prev_item -> player -> pause();
            prev_item -> player -> hide();
        }
    }
}

void NewWindow::on_addButton_clicked()
{
    while( curr_myitem_idx < myitem_vec.size() ) {
        qDebug () << "123----------------------------------";
        MyItem *curr_myitem = myitem_vec.at(curr_myitem_idx);
        layout -> addWidget(curr_myitem -> player, 0, 1);
        curr_myitem -> item -> setWhatsThis(QString::number(list_widget -> count()));
        QObject::connect(curr_myitem -> player -> mediaObject(), SIGNAL(hasVideoChanged(bool)), curr_myitem, SLOT(testSeek(bool)));
        list_widget -> insertItem(list_widget -> count(), curr_myitem -> item);
        ++curr_myitem_idx;
    }
}
