#include "tipsdialog.h"
#include "ui_tipsdialog.h"

TipsDialog::TipsDialog(bool* flg, QString text, int page, QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::TipsDialog), flags(flg)
{
    ui->setupUi(this);
    this->setWindowFlags(Qt::FramelessWindowHint);
    if (page == 0)
    {
        ui->stackedWidget->setCurrentIndex(0);
        ui->ob_label->setText(text);
        ui->ob_label->update();
    }
    else
    {
        ui->stackedWidget->setCurrentIndex(1);
        ui->db_label->setText(text);
        qDebug() << text;
        ui->db_label->update();
    }
}

TipsDialog::~TipsDialog()
{
    delete ui;
}

void TipsDialog::paintEvent(QPaintEvent *event)
{
    QStyleOption opt;
    opt.initFrom(this);
    QPainter p(this);
    style()->drawPrimitive(QStyle::PE_Widget, &opt, &p, this);
    QBitmap bmp(this->size());
    bmp.fill();
    QPainter painter(&bmp);
    painter.setPen(Qt::NoPen);
    painter.setBrush(Qt::black);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.drawRoundedRect(bmp.rect(), 12, 12);
    setMask(bmp);
    QDialog::paintEvent(event);
}


void TipsDialog::on_sigOK_btn_clicked()
{
    *flags = true;
    this->close();
}


void TipsDialog::on_dbOk_btn_clicked()
{
    *flags = true;
    this->close();
}


void TipsDialog::on_dbCancel_btn_clicked()
{
    *flags = false;
    this->close();
}

