#include <QWidget>
class MyWidget : public QWidget
{
    Q_OBJECT
public:
    MyWidget(QWidget *parent = 0);
public slots:
    void playVideo();
};

