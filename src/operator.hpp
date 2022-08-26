#ifndef OPERATOR_H
#define OPERATOR_H

#include "myeig.hpp"

using namespace std;
using namespace myeig;

struct Op {

  virtual ~Op(){};

  virtual int arity() {
    throw runtime_error("Not implemented");
  }

  virtual string sym() {
    throw runtime_error("Not implemented");
  }

  virtual Vec apply(Mat X) {
    throw runtime_error("Not implemented");
  }

};

struct Add : Op {

  int arity() override {
    return 2;
  }

  string sym() override {
    return "+";
  }

  Vec apply(Mat X) override {
    return X.rowwise().sum();
  }

};

struct Neg : Op {

  int arity() override {
    return 1;
  }

  string sym() override {
    return "¬";
  }

  Vec apply(Mat X) override {
    return -X.col(0);
  }

};

struct Sub : Op {

  int arity() override {
    return 2;
  }

  string sym() override {
    return "-";
  }

  Vec apply(Mat X) override {
    return X.col(0)-X.col(1);
  }

};

#endif