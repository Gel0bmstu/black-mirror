QT += core
QT -= gui
QMAKE_CXXFLAGS = -std=c++14
LIBS += -I/usr/local/include -lTgBot -lboost_system -lssl -lcrypto -lpthread
TARGET = tg
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    messenger.cpp

HEADERS += \
    messenger.h

