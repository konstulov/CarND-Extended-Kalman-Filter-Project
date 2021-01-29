#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  // Calculate RMSE
  size_t n = estimations.size();
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;
  // validate inputs
  if (n != ground_truth.size() || n == 0) {
    return rmse;
  }
  
  // sum squared residuals
  for (size_t i=0; i<n; ++i) {
    // residual
    VectorXd res = estimations[i] - ground_truth[i];

    // elementwise multiplication (squared residual)
    res = res.array()*res.array();
    rmse += res;
  }

  // normalize to get the mean
  rmse /= n;

  // calculate the squared root
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  // Calculate Jacobian matrix
  MatrixXd Hj(3, 4);
  
  // Get state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // Pre-compute reused terms
  float c1 = px*px + py*py;
  float c2 = sqrt(c1);
  float c3 = c1*c2;

  // check for division by zero
  if (fabs(c3) < 0.0001) {
    cout << "CalculateJacobian() - Error - Division by Zero!" << endl;
    return Hj;
  }

  // compute the Jacobian matrix
  Hj << (px/c2), (py/c2), 0, 0,
       -(py/c1), (px/c1), 0, 0,
        py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return Hj;
}
