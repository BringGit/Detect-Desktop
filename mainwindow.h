#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include "include.h"
#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    bool eventFilter(QObject *obj, QEvent *event) override;
    void paintEvent(QPaintEvent *event) override;

private slots:
    void on_document_btn_clicked();

    void on_model_btn_clicked();

    void on_detect_btn_clicked();

    void on_nms_lineEdit_textChanged(const QString &nms);

    void on_score_lineEdit_textChanged(const QString &score);

    void on_conf_lineEdit_textChanged(const QString &arg1);

    void on_select_btn_clicked();

    void on_reselect_btn_clicked();

    void on_opencam_btn_clicked();

    void on_closecam_btn_clicked();

    void DisplayVideo();

    void on_start_btn_clicked();

    void on_stop_btn_clicked();

    void on_opengpu_btn_clicked();

    void on_closegpu_btn_clicked();

    void on_inpwid_lineEdit_textChanged(const QString &arg1);

    void on_inphei_lineEdit_textChanged(const QString &arg1);

    void on_tclsname_btn_clicked();

    void on_fclsname_btn_clicked();

    void on_tscore_btn_clicked();

    void on_fscore_btn_clicked();

    void on_classdocument_btn_clicked();

    void on_min_btn_clicked();

    void on_max_btn_clicked();

    void on_close_btn_clicked();

private:
    void create_classframe();
    void clear_classframe();
    void UpdateConfig(Config cfg);

    void Initial();
    Ui::MainWindow *ui;
    QTimer *playtimer;
    bool classInitFlag;

    // std::vector<std::string> classes{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
    std::vector<std::string> classes;
    Config det_cfg;
    DetectPool *detector;
    cv::Mat frame;

};
#endif // MAINWINDOW_H
