#ifndef OPERATORS_H
#define OPERATORS_H
#include <cassert>

#include "common.h"

#include"linear.h"
#include"embedding.h"
#include"rotaryPosEmb.h"
#include"bmm.h"



#define BLK_SIZE 16
#define NUM_THREAD 4

// include all ops

void softmax(const Matrix3D<float> &input, Matrix3D<float> &output, int dim);
void batch_Add(const Matrix3D<float> &input, const Matrix3D<float> &input2, Matrix3D<float> &output);
template <typename T>
void linear(Matrix3D<T> &a, Matrix3D<T> &b, Matrix3D<T> &c);

#endif  // OPERATORS_H
