#include <QObject>
#include <QMainWindow>
#include <QWidget>
#include <phonon/VideoPlayer>
#include <phonon/VideoWidget>
#include <phonon/MediaObject>
#include <phonon/MediaSource>
#include <phonon>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QPushButton>
#include <QUrl>
#include <QSignalMapper>

class MainWindow : public QMainWindow
{
Q_OBJECT
public:
    MainWindow(QWidget *parent = 0);
    ~MainWindow();
    //QPushButton *addFile;
    QPushButton *quit;
    QPushButton *play;
    QPushButton *pause;
    QWidget *Main;
    Phonon::VideoPlayer *player;
    void startTimeSet(float);

public slots:
    //void startVideo();
    void playVideo(int);
    void pauseVideo();
    //void seekAnimation(bool);
private:
    QString _file;
    QVector<QPushButton *> _bptrVec;
    float _startTime;
    bool _seekable;
};

