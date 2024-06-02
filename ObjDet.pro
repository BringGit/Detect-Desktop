QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# INCLUDEPATH += E:/Project/C++/Opencv/opencv/mingw64build/install/include \
#                 E:/Project/C++/Opencv/opencv/mingw64build/install/include/opencv2

# LIBS +=  E:/Project/C++/Opencv/opencv/mingw64build/install/x64/mingw/lib/libopencv_*.a
# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    DetectPool.cpp \
    displayview.cpp \
    inference.cpp \
    main.cpp \
    mainwindow.cpp \
    tipsdialog.cpp

HEADERS += \
    DetectPool.h \
    displayview.h \
    include.h \
    inference.h \
    mainwindow.h \
    tipsdialog.h

FORMS += \
    mainwindow.ui \
    tipsdialog.ui

TRANSLATIONS += \
    ObjDet_en_AS.ts \
    ObjDet_zh_CN.ts
CONFIG += lrelease
CONFIG += embed_translations

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES +=

RESOURCES += \
    res.qrc



win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../C++/Opencv/opencv/build/x64/vc16/lib/ -lopencv_world490
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../C++/Opencv/opencv/build/x64/vc16/lib/ -lopencv_world490d
else:unix: LIBS += -L$$PWD/../../C++/Opencv/opencv/build/x64/vc16/ -lopencv_world490

INCLUDEPATH += $$PWD/../../C++/Opencv/opencv/build/include
DEPENDPATH += $$PWD/../../C++/Opencv/opencv/build/include
