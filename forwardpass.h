#ifndef FORWARDPASS_H
#define FORWARDPASS_H

#include <QObject>

class SamSystem;

class ForwardPass : public QObject {
    Q_OBJECT
    SamSystem *system;
public:
    explicit ForwardPass(SamSystem *system, QObject *parent = nullptr);

signals:
    void finished(const bool &success, QString log);

public slots:
    void doWork(QString fileName);
};

#endif // FORWARDPASS_H
