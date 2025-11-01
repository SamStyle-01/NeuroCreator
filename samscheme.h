#ifndef SAMSCHEME_H
#define SAMSCHEME_H

#include "pch.h"

class SamField;
class SamSystem;

struct DeviceButton : public QPushButton {
    DeviceButton(QString text, QWidget* parent, cl_device_id index);
    cl_device_id index;
};

class SamScheme : public QFrame {
    Q_OBJECT
    QGridLayout *layout;
    SamField *field;
    SamSystem *system;
public:
    explicit SamScheme(QWidget *parent, SamSystem *system);

signals:
};

#endif // SAMSCHEME_H
