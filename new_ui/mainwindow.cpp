#include "mainwindow.h"
 
MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    main_window = new QWidget; 
    main_window->resize(1200, 800);
    layout = new QGridLayout;
    list_widget = new QListWidget;
    list_widget->setIconSize(QSize(120,120));
    list_widget->setFixedWidth(400);
    list_widget->insertItem(0, new QListWidgetItem(QIcon(QString("./img/0.jpg")), "caterpillar"));
    list_widget->insertItem(1, new QListWidgetItem(QIcon("./img/1.png"), "momor"));
    list_widget->insertItem(2, new QListWidgetItem(QIcon("./img/2.png"), "bush"));

    layout->addWidget(list_widget, 0, 0);

    stacked_layout = new QStackedLayout;
    layout->addLayout(stacked_layout, 0, 1);

    stacked_layout->addWidget(new QLabel("<h1><font color=blue>caterpillar</font></h1>"));
    stacked_layout->addWidget(new QPushButton("momor"));
    stacked_layout->addWidget(new QTextEdit);

    QObject::connect(list_widget, SIGNAL(currentRowChanged(int)),
            stacked_layout, SLOT(setCurrentIndex(int)));

    main_window->setLayout(layout);
    main_window->show();

    add = new QPushButton("add");
    layout->addWidget(add, 1, 0);

    connect(add, SIGNAL(clicked()), this, SLOT(on_addButton_clicked()));
}
 
MainWindow::~MainWindow() {}
 
void MainWindow::on_addButton_clicked()
{
    //QDynamicButton *button = new QDynamicButton(this); 
    //button->setText("new button" + QString::number(button->getID()));
    //layout->addWidget(button);
    //button->show();
    //connect(button, SIGNAL(clicked()), this, SLOT(slotGetNumber()));
    QListWidgetItem *item = new QListWidgetItem(QIcon("./img/0.jpg"), "showww");
    list_widget->insertItem(3, item);
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
