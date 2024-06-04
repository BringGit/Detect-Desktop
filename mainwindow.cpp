#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "tipsdialog.h"
#include <QFile>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    setWindowIcon(QIcon(QStringLiteral(":/assets/icon.png")));
    ui->setupUi(this);
    Initial();
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::paintEvent(QPaintEvent *event)
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
    QMainWindow::paintEvent(event);
}

void MainWindow::Initial()
{
    Config cfg;
    cfg.onnxPath = "";
    cfg.inputSize = cv::Size(640, 640);
    cfg.classfile = "";
    cfg.runOnGPU = false;
    cfg.useCamera = false;
    cfg.runOnOpenvino = true;
    classInitFlag = false;

    detector = new DetectPool(cfg);
    detector->GetClasses(classes);
    ui->tableWidget->setRowCount(classes.size());
    ui->tableWidget->setColumnCount(1);

    ui->tableWidget->horizontalHeader()->setStretchLastSection(true);
    ui->tableWidget->horizontalHeader()->hide();
    ui->tableWidget->verticalHeader()->hide();
    ui->model_lineEdit->setText(QString(cfg.onnxPath.c_str()));
    ui->nms_lineEdit->setText("0.50");
    ui->score_lineEdit->setText("0.45");
    ui->conf_lineEdit->setText("0.25");
    ui->inpwid_lineEdit->setText(QString::asprintf("%d", cfg.inputSize.width));
    ui->inphei_lineEdit->setText(QString::asprintf("%d", cfg.inputSize.height));
    ui->classpath_lineEdit->setText(cfg.classfile.c_str());
    ui->tclsname_btn->setChecked(true);
    ui->tscore_btn->setChecked(true);
    ui->closecam_btn->setChecked(true);
    ui->closegpu_btn->setChecked(true);
    ui->openov_btn->setChecked(true);
    ui->information_label->setText(tr("推理时间:  ms     检测目标数:"));
    create_classframe();

    playtimer = new QTimer();
    playtimer->setInterval(1);
    connect(playtimer, &QTimer::timeout,this, &MainWindow::DisplayVideo);
}

void MainWindow::create_classframe()
{
    ui->tableWidget->setRowCount(0);
    ui->tableWidget->setRowCount(classes.size());
    int i = 0;
    for(std::string s : classes)
    {
        QCheckBox* cbox = new QCheckBox();

        cbox->setObjectName(QString("class%1").arg(i));
        cbox->setText(s.c_str());
        cbox->setCheckState(Qt::Checked);
        cbox->installEventFilter(this);
        ui->tableWidget->setCellWidget(i, 0, cbox);
        i++;
    }
}

bool MainWindow::eventFilter(QObject *obj, QEvent *event)
{
    if (event->type() == QEvent::MouseButtonPress && obj->objectName().contains("class"))
    {

        QCheckBox *cbx = static_cast<QCheckBox*>(obj);
        if (cbx->isChecked())
        {
            detector->UpdateMap(QString(obj->objectName().back()).toInt(), false);
            cbx->setChecked(false);
        }
        else
        {
            detector->UpdateMap(QString(obj->objectName().back()).toInt(), true);
            cbx->setChecked(true);
        }

        return true;
    }
    return QObject::eventFilter(obj, event);
}

void MainWindow::mousePressEvent(QMouseEvent *event)
{
    /*当鼠标左键点击时.*/
    if (event->button() == Qt::LeftButton)
    {
        m_move = true;
        /*记录鼠标的世界坐标.*/
        m_startPoint = event->globalPos();
        /*记录窗体的世界坐标.*/
        m_windowPoint = this->frameGeometry().topLeft();
    }
}

void MainWindow::mouseMoveEvent(QMouseEvent *event)
{
    if (event->buttons() & Qt::LeftButton)
    {
        /*移动中的鼠标位置相对于初始位置的相对位置.*/
        QPoint relativePos = event->globalPos() - m_startPoint;
        /*然后移动窗体即可.*/
        this->move(m_windowPoint + relativePos );
    }
}

void MainWindow::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
    {
        /*改变移动状态.*/
        m_move = false;
    }
}
void MainWindow::UpdateConfig(Config cfg)
{
    det_cfg = cfg;
}

void MainWindow::on_document_btn_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this,"选择文件","./");
    ui->path_lineEdit->setText(filename);
    QMimeDatabase mimeDatabase;
    QMimeType mimeType = mimeDatabase.mimeTypeForFile(filename);

    if (mimeType.isValid()) {
        qDebug() << "File MIME Type: " << mimeType.name();
    } else {
        qDebug() << "Unknown MIME Type for file: " << filename;
    }
    if (mimeType.name().startsWith("image/"))
    {
        frame = cv::imread(filename.toStdString());
        QImage qImage(detector->CV2QT(frame),frame.cols,frame.rows,frame.step,QImage::Format_RGB888);
        QGraphicsScene *scene = new QGraphicsScene();
        ui->graphicsView->setScene(scene);
        ui->graphicsView->show();
        scene->addPixmap(QPixmap::fromImage(qImage));
    }
    else if (mimeType.name().startsWith("video/"))
    {
        detector->OpenCamera(frame, 0, true, filename.toStdString());
        QImage qImage(detector->CV2QT(frame),frame.cols,frame.rows,frame.step,QImage::Format_RGB888);
        QGraphicsScene *scene = new QGraphicsScene();
        ui->graphicsView->setScene(scene);
        ui->graphicsView->show();
        scene->addPixmap(QPixmap::fromImage(qImage));
    }
}


void MainWindow::on_model_btn_clicked()
{
    QString s = QFileDialog::getOpenFileName(this,"选择文件","./", tr("模型文件(*onnx, *xml)"));
    ui->model_lineEdit->setText(s);

    Config cfg;
    cfg.onnxPath = s.toStdString();
    if (s.contains(".xml"))
    {
        cfg.runOnOpenvino = true;
    }
    detector->UpdateConfig(cfg);
}

void MainWindow::on_detect_btn_clicked()
{
    double t;
    int ndet;
    bool flag=false;
    TipsDialog *tip = new TipsDialog(&flag, tr("请确认配置设置是否正确!"), 1);
    tip->exec();
    if (flag)
    {

        cv::Mat output = detector->GetOutPut(frame, t, ndet);

        // ui->information_label->setText(QString::asprintf("推理时间: %.3f ms     检测目标数: %d", t, ndet));
        ui->information_label->setText(tr(QString("推理时间: %1 ms     检测目标数: %2").arg(t).arg(ndet).toStdString().c_str()));
        QGraphicsScene *scene = new QGraphicsScene();
        QImage qImage((unsigned char*)output.data,output.cols,output.rows,output.step,QImage::Format_RGB888);
        ui->graphicsView->setScene(scene);
        ui->graphicsView->show();
        scene->addPixmap(QPixmap::fromImage(qImage));
    }
}


void MainWindow::on_nms_lineEdit_textChanged(const QString &nms)
{

    Config cfg;
    cfg = detector->GetConfig();
    cfg.onnxPath = "";
    cfg.classfile = "";
    cfg.NMSThreshold = nms.toFloat();
    detector->UpdateConfig(cfg);
}

void MainWindow::on_score_lineEdit_textChanged(const QString &score)
{
    Config cfg;

    cfg = detector->GetConfig();
    cfg.onnxPath = "";
    cfg.classfile = "";
    cfg.ScoreThreshold = score.toFloat();
    detector->UpdateConfig(cfg);
}


void MainWindow::on_conf_lineEdit_textChanged(const QString &conf)
{
    Config cfg;
    cfg = detector->GetConfig();
    cfg.onnxPath = "";
    cfg.classfile = "";
    cfg.ConfidenceThreshold = conf.toFloat();
    detector->UpdateConfig(cfg);
}


void MainWindow::on_select_btn_clicked()
{

    for(int i = 0; i < classes.size(); i++)
    {
        QWidget *wid = (QWidget *)ui->tableWidget->cellWidget(i, 0);
        QCheckBox* cbx = qobject_cast<QCheckBox*>(wid);
        cbx->setChecked(true);
        detector->UpdateMap(QString(cbx->objectName().back()).toInt(), true);
    }
}


void MainWindow::on_reselect_btn_clicked()
{
    for(int i = 0; i < classes.size(); i++)
    {
        QWidget *wid = (QWidget *)ui->tableWidget->cellWidget(i, 0);
        QCheckBox* cbx = qobject_cast<QCheckBox*>(wid);
        if (cbx->isChecked())
        {
            cbx->setChecked(false);
            detector->UpdateMap(QString(cbx->objectName().back()).toInt(), false);
        }
        else
        {
            cbx->setChecked(true);
            detector->UpdateMap(QString(cbx->objectName().back()).toInt(), true);
        }

    }
}


void MainWindow::on_opencam_btn_clicked()
{
    detector->OpenCamera(frame, 0, false);
    QImage qImage(detector->CV2QT(frame),frame.cols,frame.rows,frame.step,QImage::Format_RGB888);
    QGraphicsScene *scene = new QGraphicsScene();
    ui->graphicsView->setScene(scene);
    ui->graphicsView->show();
    scene->addPixmap(QPixmap::fromImage(qImage));
}


void MainWindow::on_closecam_btn_clicked()
{
    detector->CloseCamera();
}


void MainWindow::DisplayVideo()
{
    double t = 0.0;
    int ndet = 0;

    if (detector->SendFrame(frame, t, ndet))
    {
        ui->information_label->setText(QString::asprintf("推理时间: %.3f ms     检测目标数: %d", t, ndet));
        QImage qImage(detector->CV2QT(frame),frame.cols,frame.rows,frame.step,QImage::Format_RGB888);
        QGraphicsScene *scene = new QGraphicsScene();
        ui->graphicsView->setScene(scene);
        ui->graphicsView->show();
        scene->addPixmap(QPixmap::fromImage(qImage));
    }
    else
    {
        playtimer->stop();
    }
}


void MainWindow::on_start_btn_clicked()
{
    playtimer->start();
}


void MainWindow::on_stop_btn_clicked()
{
    playtimer->stop();
}


void MainWindow::on_opengpu_btn_clicked()
{
    Config cfg;
    cfg = detector->GetConfig();
    cfg.runOnGPU = true;
    detector->UpdateConfig(cfg);
}


void MainWindow::on_closegpu_btn_clicked()
{
    Config cfg;
    cfg = detector->GetConfig();
    cfg.runOnGPU = false;
    detector->UpdateConfig(cfg);
}


void MainWindow::on_inpwid_lineEdit_textChanged(const QString &width)
{
    Config cfg;
    cfg = detector->GetConfig();
    cfg.inputSize = cv::Size(width.toInt(), cfg.inputSize.height);
    detector->UpdateConfig(cfg);
}


void MainWindow::on_inphei_lineEdit_textChanged(const QString &height)
{
    Config cfg;
    cfg = detector->GetConfig();
    cfg.inputSize = cv::Size(cfg.inputSize.width, height.toInt());
    detector->UpdateConfig(cfg);
}


void MainWindow::on_tclsname_btn_clicked()
{
    Config cfg;
    cfg = detector->GetConfig();
    cfg.displayClassName = true;
    detector->UpdateConfig(cfg);
}


void MainWindow::on_fclsname_btn_clicked()
{
    Config cfg;
    cfg = detector->GetConfig();
    cfg.displayClassName = false;
    detector->UpdateConfig(cfg);
}


void MainWindow::on_tscore_btn_clicked()
{
    Config cfg;
    cfg = detector->GetConfig();
    cfg.displayScore = true;
    detector->UpdateConfig(cfg);
}


void MainWindow::on_fscore_btn_clicked()
{
    Config cfg;
    cfg = detector->GetConfig();
    cfg.displayScore = false;
    detector->UpdateConfig(cfg);
}


void MainWindow::on_classdocument_btn_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this,"选择文件","./", tr("文本文件(*txt)"));
    ui->classpath_lineEdit->setText(filename);
    Config cfg;
    cfg = detector->GetConfig();
    cfg.classfile = filename.toStdString();
    detector->UpdateConfig(cfg);
    detector->GetClasses(classes);
    create_classframe();
}


void MainWindow::on_min_btn_clicked()
{
    this->showMinimized();
}


void MainWindow::on_max_btn_clicked()
{
    if (this->isMaximized())
        this->showNormal();
    else
        this->showMaximized();
}


void MainWindow::on_close_btn_clicked()
{
    this->close();
}


void MainWindow::on_openov_btn_clicked()
{
    Config cfg;
    cfg = detector->GetConfig();
    cfg.runOnOpenvino = true;
    detector->UpdateConfig(cfg);
}


void MainWindow::on_closeov_btn_clicked()
{
    Config cfg;
    cfg = detector->GetConfig();
    cfg.runOnOpenvino = false;
    detector->UpdateConfig(cfg);
}

