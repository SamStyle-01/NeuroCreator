#ifndef SAMSCHEME_H
#define SAMSCHEME_H

#include "pch.h"

class SamField;
class SamSystem;
class SamView;
class SamButtonsGroup;

struct DeviceButton : public QPushButton {
    DeviceButton(QString text, QWidget* parent, cl_device_id index);
    cl_device_id index;
};

class SamScheme : public QFrame {
    Q_OBJECT
    QGridLayout *layout;
    SamField *field;
    SamSystem *system;
    SamView* view;
    QTextEdit* output_field;
    SamButtonsGroup* devices;
    bool manual_input_now;
public:
    explicit SamScheme(SamView *parent, SamSystem *system);
    void set_output_field(QString values);
    QVector<QPushButton*> get_devices();

signals:
};

#endif // SAMSCHEME_H
