#include "displayview.h"

DisplayView::DisplayView(QWidget *parent) :
    QGraphicsView(parent)
{
    setDragMode(QGraphicsView::ScrollHandDrag);

    QGraphicsPixmapItem *pixmapItem = new QGraphicsPixmapItem(QPixmap(":/images/my_image.png"));
    pixmapItem->setTransformationMode(Qt::SmoothTransformation);

    QGraphicsScene *scene = new QGraphicsScene();
    scene->addItem(pixmapItem);
    setScene(scene);
}

void DisplayView::wheelEvent(QWheelEvent *event)
{
    if(event->angleDelta().y() > 0)
        scale(1.25, 1.25);
    else
        scale(0.8, 0.8);
}

void DisplayView::keyPressEvent(QKeyEvent *event)
{
    if(event->key() == Qt::Key_Left)
        rotate(1);
    else if(event->key() == Qt::Key_Right)
        rotate(-1);
}
