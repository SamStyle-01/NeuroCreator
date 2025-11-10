QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets svg charts

CONFIG += c++17 precompile_header

PRECOMPILED_HEADER = \
    pch.h

INCLUDEPATH += $$PWD/include \
               $$PWD/include/CL \
               $$PWD/include/libclew

LIBS += -L$$PWD/libs -lclew

SOURCES += \
    backward.cpp \
    dataframe.cpp \
    forwardpass.cpp \
    main.cpp \
    sambuttonsgroup.cpp \
    samchart.cpp \
    samfield.cpp \
    sammodel.cpp \
    samscheme.cpp \
    samsystem.cpp \
    samtraining.cpp \
    samview.cpp

HEADERS += \
    backward.h \
    dataframe.h \
    forwardpass.h \
    pch.h \
    sambuttonsgroup.h \
    samchart.h \
    samfield.h \
    sammodel.h \
    samscheme.h \
    samsystem.h \
    samtraining.h \
    samview.h \
    include/libclew/ocl_init.h

DEFINES += CL_TARGET_OPENCL_VERSION=220

QMAKE_CXXFLAGS += -Wno-deprecated-declarations
QMAKE_CXXFLAGS += -D__cpuidex=__cpuidex_disabled

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
