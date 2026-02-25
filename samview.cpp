#include "samview.h"
#include "samscheme.h"
#include "samtraining.h"
#include "samsystem.h"

SamView::SamView(QWidget *parent) : QStackedWidget{parent} {
    this->setWindowTitle("Конструктор Нейросетей");
    this->isFullScreen = false;
    this->setWindowIcon(QIcon("./icon.ico"));
    this->state = State::SCHEME;
}

void SamView::init(SamScheme* scheme, SamTraining* training, SamSystem* system) {
    this->scheme = scheme;
    this->training = training;
    this->addWidget(this->scheme);
    this->addWidget(this->training);
    this->resize(this->scheme->width(), this->scheme->height());
    this->setCurrentIndex(0);
    this->system = system;
}

void SamView::keyPressEvent(QKeyEvent *event) {
    if (event->key() == Qt::Key_F11) {
        if (this->isFullScreen) {
            this->showNormal();
        }
        else {
            this->showFullScreen();
        }
        this->isFullScreen = !this->isFullScreen;
    }
    QStackedWidget::keyPressEvent(event);
}
