#include <iostream>

using namespace std;

#include <ctime>
// Eigen core part
#include <Eigen/Core>

// Algebraic operations on dense matrices (inverse, eigenvalues, etc.)
#include <Eigen/Dense>

using namespace Eigen;

#define MATRIX_SIZE 50

/****************************
*This program demonstrates the use of basic types of Eigen
****************************/

int main(int argc, char **argv) {
  // All vectors and matrices in Eigen are Eigen :: Matrix, which is a template class. Its first three parameters are: data type, row, column
  // Declare a 2 * 3 float matrix
  Matrix<float, 2, 3> matrix_23;

  // At the same time, Eigen provides many built-in types through typedef, but the bottom layer is still Eigen :: Matrix
  // For example, Vector3d is essentially Eigen :: Matrix <double, 3, 1>, which is a three-dimensional vector
  Vector3d v_3d;
  // It's the same
  Matrix<float, 3, 1> vd_3d;

  // Matrix3d 实质上是 Eigen::Matrix<double, 3, 3>
  Matrix3d matrix_33 = Matrix3d::Zero(); //Initialized to zero

  // If you are not sure about the size of the matrix, you can use a dynamically sized matrix
  Matrix<double, Dynamic, Dynamic> matrix_dynamic;
  // Simpler
  MatrixXd matrix_x;
  // There are many more of this type, we do not list them one by one

  // The following is the operation of Eigen array
  // Input data (initialization)
  matrix_23 << 1, 2, 3, 4, 5, 6;
  // Output
  cout << "matrix 2x3 from 1 to 6: \n" << matrix_23 << endl;

  // Use () to access the elements in the matrix
  cout << "print matrix 2x3: " << endl;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) cout << matrix_23(i, j) << "\t";
    cout << endl;
  }

  //Multiplying matrix and vector (actually still matrix and matrix)

  v_3d << 3, 2, 1;
  vd_3d << 4, 5, 6;

  // But in Eigen you ca n’t mix two different types of matrices, it ’s wrong like this

  // Matrix<double, 2, 1> result_wrong_type = matrix_23 * v_3d;
  // Should be converted explicitly

  Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
  cout << "[1,2,3;4,5,6]*[3,2,1]=" << result.transpose() << endl;

  Matrix<float, 2, 1> result2 = matrix_23 * vd_3d;
  cout << "[1,2,3;4,5,6]*[4,5,6]: " << result2.transpose() << endl;

  // Also you can't get the matrix dimensions wrong

  // Try to cancel the comment below and see what error Eigen will report

  // Eigen::Matrix<double, 2, 3> result_wrong_dimension = matrix_23.cast<double>() * v_3d;

  // Some matrix operations

  // Four operations will not be demonstrated, just use +-* /.

  matrix_33 = Matrix3d::Random();      // Random number matrix
  cout << "random matrix: \n" << matrix_33 << endl;
  cout << "transpose: \n" << matrix_33.transpose() << endl;      // Transpose
  cout << "sum: " << matrix_33.sum() << endl;            // Elements and
  cout << "trace: " << matrix_33.trace() << endl;          // trace
  cout << "times 10: \n" << 10 * matrix_33 << endl;               // Multiply
  cout << "inverse: \n" << matrix_33.inverse() << endl;        // Reverse
  cout << "det: " << matrix_33.determinant() << endl;    // Determinant

  // Eigenvalues
  // Real symmetric matrix can ensure the success of diagonalization

  SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
  cout << "Eigen values = \n" << eigen_solver.eigenvalues() << endl;
  cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;

  // Solving equations
  // We solve the equation matrix_NN * x = v_Nd
  // The size of N is defined in the preceding macro, which is generated by a random number
  // Direct inversion is naturally the most direct, but the inverse calculation is large


  Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN
      = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
  matrix_NN = matrix_NN * matrix_NN.transpose();  // Guaranteed positive semidefinite
  Matrix<double, MATRIX_SIZE, 1> v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);

  clock_t time_stt = clock(); // Timing
  // Direct inversion
  Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
  cout << "time of normal inverse is "
       << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
  cout << "x = " << x.transpose() << endl;

  // Usually matrix decomposition, such as QR decomposition, will be much faster
  time_stt = clock();
  x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
  cout << "time of Qr decomposition is "
       << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
  cout << "x = " << x.transpose() << endl;

  // For positive definite matrices, you can also use cholesky decomposition to solve equations
  time_stt = clock();
  x = matrix_NN.ldlt().solve(v_Nd);
  cout << "time of ldlt decomposition is "
       << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
  cout << "x = " << x.transpose() << endl;

  return 0;
}