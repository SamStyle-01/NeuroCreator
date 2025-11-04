#ifndef SAMVIEW_H
#define SAMVIEW_H

#include "pch.h"

class SamScheme;
class SamTraining;
class SamSystem;

enum class State {
    SCHEME,
    TRAINING
};

class SamView : public QStackedWidget {
    Q_OBJECT
    bool isFullScreen;
    SamScheme* scheme;
    SamTraining* training;
    State state;
    SamSystem* system;
public:
    SamView(QWidget *parent = nullptr);
    void keyPressEvent(QKeyEvent *event) override;
    void init(SamScheme* scheme, SamTraining* training, SamSystem *system);
};

#endif // SAMVIEW_H
