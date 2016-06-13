#ifndef MAINWINDOW_H
#define MAINWINDOW_H
 
#include <QMainWindow>
#include <QGridLayout> 
#include <QStackedLayout>
#include <QListWidget>
#include <QLabel>
#include <QTextEdit>
/* My Includes */

#include <qdynamicbutton.h>
 
//namespace Ui {
//class MainWindow;
//}
 
class MainWindow : public QMainWindow
{
    Q_OBJECT
 
public:
    QWidget *main_window;
    QGridLayout *layout;
    QListWidget *list_widget;
    QStackedLayout *stacked_layout;
    QPushButton *add;
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
 
private slots:
    void on_addButton_clicked();    // СЛОТ-обработчик нажатия кнопки добавления
    void on_deleteButton_clicked(); // СЛОТ-обработчик нажатия кнопки удаления
    void slotGetNumber();           // СЛОТ для получения номера нажатой динамической кнопки
 
private:
    QWidget *_parent;
};
 
#endif // MAINWINDOW_H
