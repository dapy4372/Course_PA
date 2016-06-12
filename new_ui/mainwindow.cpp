#include "mainwindow.h"
 
MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    Main = new QWidget; 
    layout = new QVBoxLayout;
    add = new QPushButton("add");
    layout->addWidget(add);
    Main->setLayout(layout);
    Main->show();
    connect(add, SIGNAL(clicked()), this, SLOT(on_addButton_clicked()));
}
 
MainWindow::~MainWindow() {}
 
void MainWindow::on_addButton_clicked()
{
    QDynamicButton *button = new QDynamicButton(this); 
    button->setText("new button" + QString::number(button->getID()));
    layout->addWidget(button);
    button->show();
    connect(button, SIGNAL(clicked()), this, SLOT(slotGetNumber()));
}
 
/* Метод для удаления динамической кнопки по её номеру
 * */
void MainWindow::on_deleteButton_clicked()
{
    for(int i = 0; i < layout->count(); i++){
        QDynamicButton *button = qobject_cast<QDynamicButton*>(layout->itemAt(i)->widget());
      //  if(button->getID() == parent->lineEdit->text().toInt()){
            button->hide();
            delete button;
        //}
    }
}
 
void MainWindow::slotGetNumber()
{
    QDynamicButton *button = (QDynamicButton*) sender();
    //parent->lineEdit->setText(QString::number(button->getID()));
}
