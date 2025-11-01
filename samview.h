#ifndef SAMVIEW_H
#define SAMVIEW_H

#include "pch.h"

class SamScheme;
class SamAnalysis;
class SamSystem;

enum class State {
    SCHEME,
    ANALYSIS
};

class SamView : public QStackedWidget {
    Q_OBJECT
    bool isFullScreen;
    SamScheme* scheme;
    SamAnalysis* analysis;
    State state;
    SamSystem* system;
public:
    SamView(QWidget *parent = nullptr);
    void keyPressEvent(QKeyEvent *event) override;
    void init(SamScheme* scheme, SamAnalysis* analysis, SamSystem *system);
};

#endif // SAMVIEW_H
