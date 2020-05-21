#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.hpp"

using namespace std;
using namespace Eigen;

/// This program demonstrates the basic usage of sophus

int main(int argc, char **argv) {

  // Rotation matrix rotated 90 degrees along the Z axis
  Matrix3d R = AngleAxisd(M_PI / 2, Vector3d(0, 0, 1)).toRotationMatrix();
  // Or quaternion
  Quaterniond q(R);
  Sophus::SO3d SO3_R(R); // Sophus::SO(3) can be constructed directly from the rotation matrix
  Sophus::SO3d SO3_q(q); // can also be constructed from a rotation vector
  // 二者是等价的
  cout << "SO(3) from matrix:\n" << SO3_R.matrix() << endl;
  cout << "SO(3) from quaternion:\n" << SO3_q.matrix() << endl;
  cout << "they are equal" << endl;

  // Use the logarithmic map to get its Lie algebra
  Vector3d so3 = SO3_R.log();
  cout << "so3 = " << so3.transpose() << endl;
  // hat is vector to antisymmetric matrix
  cout << "so3 hat=\n" << Sophus::SO3d::hat(so3) << endl;
  // Relative, vee is the objection vector
  cout << "so3 hat vee= " << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose() << endl;

  // Update of the incremental disturbance model
  Vector3d update_so3(1e-4, 0, 0); //assuming the update is so much
  Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
  cout << "SO3 updated = \n" << SO3_updated.matrix() << endl;

  cout << "*******************************" << endl;
  // The operation of SE(3) is similar
  Vector3d t(1, 0, 0);           // translate 1 along the X axis
  Sophus::SE3d SE3_Rt(R, t);           // Construct SE(3) from R, t
  Sophus::SE3d SE3_qt(q, t);            // Construct SE(3) from q,t
  cout << "SE3 from R,t= \n" << SE3_Rt.matrix() << endl;
  cout << "SE3 from q,t= \n" << SE3_qt.matrix() << endl;
  // The Lie algebra se(3) is a six-dimensional vector, which is convenient for typedef first.
  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  Vector6d se3 = SE3_Rt.log();
  cout << "se3 = " << se3.transpose() << endl;
  //Observe the output, you will find that in Sophus, the translation of se(3) is in front and the rotation is in the back.
  // Same, there are two operators of hat and vee
  cout << "se3 hat = \n" << Sophus::SE3d::hat(se3) << endl;
  cout << "se3 hat vee = " << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose() << endl;

  //  Finally, demonstrate the update
  Vector6d update_se3; //update volume
  update_se3.setZero();
  update_se3(0, 0) = 1e-4;
  Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
  cout << "SE3 updated = " << endl << SE3_updated.matrix() << endl;

  return 0;
}