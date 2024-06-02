#ifndef TIPSDIALOG_H
#define TIPSDIALOG_H
#include "include.h"
#include <QDialog>

namespace Ui {
class TipsDialog;
}

class TipsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit TipsDialog(bool* flag, QString text, int page, QWidget *parent = nullptr);
    ~TipsDialog();

private slots:
    void on_sigOK_btn_clicked();

    void on_dbOk_btn_clicked();

    void on_dbCancel_btn_clicked();

private:
    void paintEvent(QPaintEvent *event);
    Ui::TipsDialog *ui;
    bool* flags;
};

#endif // TIPSDIALOG_H
