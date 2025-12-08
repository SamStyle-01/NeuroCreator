#ifndef BACKPROP_H
#define BACKPROP_H

#include <QObject>
#include "pch.h"
#include <QtGlobal>


class SamSystem;
class DataFrame;

class SamArray {
public:
    cl_mem memory = nullptr;
    size_t size = 0;
    bool is_inited = false;

    SamArray() = default;

    SamArray(cl_mem mem, size_t s) noexcept;

    SamArray(const SamArray&);
    SamArray& operator=(const SamArray&);

    SamArray(SamArray&& other) noexcept;

    SamArray& operator=(SamArray&& other) noexcept;

    ~SamArray();

    void clear();

private:
    void release() noexcept;
};

Q_DECLARE_TYPEINFO(SamArray, Q_MOVABLE_TYPE);

class BackPropagation : public QObject {
    Q_OBJECT
    SamSystem *system;
public:
    explicit BackPropagation(SamSystem *system, QObject *parent = nullptr);

signals:
    void finished(const bool &success, QString log);
    void epoch_done(float train_loss, float valid_loss);

public slots:
    void doWork(cl_context& context);
};

#endif // BACKPROP_H
