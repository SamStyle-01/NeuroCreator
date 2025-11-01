#ifndef SAMANALYSIS_H
#define SAMANALYSIS_H

#include <QWidget>

class SamSystem;

class SamAnalysis : public QWidget {
    Q_OBJECT
    QGridLayout* layout;
    SamSystem *system;
public:
    explicit SamAnalysis(QWidget *parent, SamSystem *system);

signals:
};

#endif // SAMANALYSIS_H
