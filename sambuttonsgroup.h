#ifndef SAMBUTTONSGROUP_H
#define SAMBUTTONSGROUP_H

#include "pch.h"
#include <functional>

class SamButtonsGroup : public QFrame {
    Q_OBJECT
    QVBoxLayout* layout;
    QLabel* label;
public:
    explicit SamButtonsGroup(QWidget *parent = nullptr);
    QVector<QPushButton*> btns;
    void addBtn(QPushButton* btn, std::function<void()> fn);
    void setLabel(QLabel *label, QString color);
    void addStretch();
signals:
};

#endif // SAMBUTTONSGROUP_H
