#ifndef TRAINING_H
#define TRAINING_H

#include "pch.h"

class SamField;
class SamSystem;
class SamView;

class SamTraining : public QFrame {
    Q_OBJECT
    QGridLayout *layout;
    QFrame *field;
    SamSystem *system;
    SamView* view;
public:
    explicit SamTraining(SamView *parent, SamSystem *system);

signals:
};

#endif // SAMSCHEME_H
