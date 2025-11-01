#include "samanalysis.h"
#include "samsystem.h"

SamAnalysis::SamAnalysis(QWidget *parent, SamSystem *system) : QWidget{parent} {
    int height = 930;
    int width = 1300;
    this->setMinimumSize(width, height);
    this->setStyleSheet("background-color: #F5EBE0;");
    this->layout = new QGridLayout(this);
    this->layout->setContentsMargins(0, 0, 0, 0);
    this->system = system;
}
