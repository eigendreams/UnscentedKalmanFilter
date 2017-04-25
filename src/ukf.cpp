#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;
	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;

	// initial state vector
	x_ = VectorXd(5);
	// initial covariance matrix
	P_ = MatrixXd(5, 5);

	// Laser measurement noise standard deviation position1 in m
	std_laspx_ = 0.15;
	// Laser measurement noise standard deviation position2 in m
	std_laspy_ = 0.15;
	// Radar measurement noise standard deviation radius in m
	std_radr_ = 0.3;
	// Radar measurement noise standard deviation angle in rad
	std_radphi_ = 0.03;
	// Radar measurement noise standard deviation radius change in m/s
	std_radrd_ = 0.3;

	/**
  DONE:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
	 */

	is_initialized_ = false;
	time_us_ = 0;

	// Obtained by trial and error
	std_a_ = 7;
	std_yawdd_ = 1.2;

	x_ << 	0, 0, 0, 0, 0;
	// From Udacity example
	P_ <<  	0.0043,    -0.0013,     0.0030,   -0.0022,   -0.0020,
		   -0.0013,     0.0077,    	0.0011,    0.0071,    0.0060,
			0.0030,    	0.0011,    	0.0054,    0.0007,    0.0008,
		   -0.0022,     0.0071,    	0.0007,    0.0098,    0.0100,
		   -0.0020,     0.0060,    	0.0008,    0.0100,    0.0123;

	n_x_ = 5;
	n_aug_ = 7;
	lambda_ = 3 - n_aug_;
	weights_ = VectorXd(1 + 2 * n_aug_);

	NIS_radar_ = 0;
	NIS_laser_ = 0;

	Xsig_pred_ = MatrixXd(n_x_, 1 + 2 * n_aug_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
	/**
  DONE:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
	 */
	if (!is_initialized_)
	{
		cout << "EKF: " << endl;

		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			/**
		      	  	  Convert radar from polar to cartesian coordinates and initialize state.
			 */

			float rho  = meas_package.raw_measurements_[0];
			float phi  = meas_package.raw_measurements_[1];
			float drho = meas_package.raw_measurements_[2];

			float x = rho * cos(phi);
			float y = rho * sin(phi);
			float v = drho;

			x_ << x, y, v, phi, 0;

			// In the spirit of the UKF, we will not get a Jacobian, and so
			// we cannot get a better initialization at this point!
		}
		if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			/**
		      Initialize state.
			 */

			float  x = meas_package.raw_measurements_[0];
			float  y = meas_package.raw_measurements_[1];

			x_ << x, y, 0, 0, 0;

			// In the spirit of the UKF, we will not get a Jacobian, and so
			// we cannot get a better initialization at this point!
		}

		time_us_ = meas_package.timestamp_;
		is_initialized_ = true;
		return;
	}

	double dt = (meas_package.timestamp_ - time_us_) / 1.0e6;

	// We can get better predictions by a small step update
	while ( dt > 0.1 )
	{
		Prediction(0.1);
		dt -= 0.1;
	}
	Prediction(dt);

	if ( meas_package.sensor_type_ == MeasurementPackage::LASER )
	{
		UpdateLidar(meas_package);
	}
	if ( meas_package.sensor_type_ == MeasurementPackage::RADAR )
	{
		UpdateRadar(meas_package);
	}

	time_us_ = meas_package.timestamp_;

	// print the output
	//cout << "x_ = " << x_ << endl;
	//cout << "P_ = " << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
	/**
  DONE:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
	 */

	float px   = x_(0);
	float py   = x_(1);
	float v    = x_(2);
	float yaw  = x_(3);
	float dyaw = x_(4);

	//cout << "X = \n" << x_ << endl;

	// state and cov matrix expansion
	VectorXd x_pred_aug = VectorXd(n_aug_);
	x_pred_aug << px, py, v, yaw, dyaw, 0, 0;
	//
	MatrixXd P_pred = P_;
	P_pred.conservativeResize(n_aug_, n_aug_);
	P_pred.col(5).setZero();
	P_pred.col(6).setZero();
	P_pred.row(5).setZero();
	P_pred.row(6).setZero();
	// Set Q
	P_pred(5, 5) = std_a_ * std_a_;
	P_pred(6, 6) = std_yawdd_ * std_yawdd_;
	// Obtain roots of cov matrix, dunno if we should use Cholesky, as 7 is a low dimensionality
	MatrixXd P_root( P_pred.llt().matrixL() );

	//cout << "P = \n" << P_pred << endl;
	//cout << "P_root = \n" << P_root << endl;

	// Generate the sigma points
	MatrixXd Xsig_pred_aug;
	Xsig_pred_aug = MatrixXd(n_aug_, 1 + 2 * n_aug_);
	Xsig_pred_aug.col(0) = x_pred_aug;
	for ( int idx = 0; idx < n_aug_; idx ++ )
	{
		Xsig_pred_aug.col(1 + 2 * idx) 	   = x_pred_aug + sqrt(lambda_ + n_aug_) * P_root.col(idx);
		Xsig_pred_aug.col(1 + 2 * idx + 1) = x_pred_aug - sqrt(lambda_ + n_aug_) * P_root.col(idx);
	}

	//cout << "xsigorig = \n" << Xsig_pred_aug << endl;

	// Pass the sigma points through model
	for ( int idx = 0; idx < 1 + 2 * n_aug_; idx++ )
	{
		VectorXd point = Xsig_pred_aug.col(idx);

		float dt2 = delta_t * delta_t;

		// assign init original values to predictions
		float yaw = point(3);

		float px_pred   = point(0);
		float py_pred   = point(1);
		float v_pred    = point(2);
		float yaw_pred  = point(3);
		float dyaw_pred = point(4);
		// These are not part of the prediction, but used for it
		float va        = point(5);
		float vyawdd    = point(6);

		if ( abs(dyaw_pred) < 1e-4 )
		{
			// Assume 0 yaw rate locally, though only for calculation, old value will
			// be carried over
			px_pred   += v_pred * cos(yaw_pred) + v_pred * cos(yaw_pred) * delta_t;
			py_pred   += v_pred * sin(yaw_pred) + v_pred * sin(yaw_pred) * delta_t;
			v_pred    += 0;
			yaw_pred  += 0;
			dyaw_pred += 0;

		}
		else
		{
			// Do as always
			px_pred   += (v_pred / dyaw_pred) * ( sin(yaw_pred + dyaw_pred * delta_t) - sin(yaw_pred));
			py_pred   += (v_pred / dyaw_pred) * (-cos(yaw_pred + dyaw_pred * delta_t) + cos(yaw_pred));
			v_pred    += 0;
			yaw_pred  += dyaw_pred * delta_t;
			dyaw_pred += 0;
		}

		// add noise effect
		px_pred 	+= 0.5 * dt2 * cos(yaw) * va;
		py_pred 	+= 0.5 * dt2 * sin(yaw) * va;
		v_pred  	+= delta_t * va;
		yaw_pred 	+= 0.5 * dt2 * vyawdd;
		dyaw_pred 	+= delta_t * vyawdd;

		// normalize angles, not a good idea to do this here!
		// But it will be an issue during the measurement!
		//while ( yaw_pred > M_PI )
		//	yaw_pred = yaw_pred - 2 * M_PI;
		//while ( yaw_pred < -M_PI )
		//	yaw_pred = yaw_pred + 2 * M_PI;

		// assign to point, last values will be ignored later
		point << px_pred, py_pred, v_pred, yaw_pred, dyaw_pred, 0, 0;
		Xsig_pred_aug.col(idx) = point;
	}

	// Revert to lower dimensionality
	for ( int idx = 0; idx < 1 + 2 * n_aug_; idx++ )
	{
		VectorXd point = VectorXd(5);
		VectorXd point_aug = VectorXd(7);

		point_aug = Xsig_pred_aug.col(idx);

		point << 	point_aug(0),
					point_aug(1),
					point_aug(2),
					point_aug(3),
					point_aug(4);

		Xsig_pred_.col(idx) = point;
	}

	//cout << "xsigpred = \n" << Xsig_pred_ << endl;

	// Get middle points, first populate list of weights
	weights_.fill(0.5 / (lambda_ + n_aug_));
	weights_(0) = (lambda_ / (lambda_ + n_aug_));

	//cout << "weights = \n" << weights_.transpose() << endl;

	// Calculate new mean
	VectorXd x_pred = x_;
	x_pred << 0, 0, 0, 0, 0;
	for ( int idx = 0; idx < 1 + 2 * n_aug_; idx++ )
	{
		x_pred += weights_(idx) * Xsig_pred_.col(idx);
	}

	//while ( x_pred(3) > M_PI )
	//	x_pred(3) = x_pred(3) - 2 * M_PI;
	//while ( x_pred(3) < -M_PI )
	//	x_pred(3) = x_pred(3) + 2 * M_PI;

	// Calculate new P
	MatrixXd P_new = P_;
	P_new.setZero();
	for ( int idx = 0; idx < 1 + 2 * n_aug_; idx++ )
	{
		VectorXd x_diff =  Xsig_pred_.col(idx) - x_pred;

		while ( x_diff(3) > M_PI )
			x_diff(3) = x_diff(3) - 2 * M_PI;
		while ( x_diff(3) < -M_PI )
			x_diff(3) = x_diff(3) + 2 *M_PI;

		P_new += weights_(idx) * x_diff * x_diff.transpose();
	}

	// Set back into model
	x_ = x_pred;
	P_ = P_new;

	// For debugging
	//cout << "x_new = \n" << x_ << endl;
	//cout << "P_new = \n" << P_ << endl;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
	/**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
	 */
	MatrixXd Z_list;
	Z_list = MatrixXd(2, 1 + 2 * n_aug_);

	// From all sigma points
	for( int idx = 0; idx < 1 + 2 * n_aug_; idx++ )
	{
		VectorXd x_sigma;
		x_sigma = VectorXd(5);
		x_sigma = Xsig_pred_.col(idx);

		// The laser model is just [I_2,2, O_2,3]
		VectorXd x_model;
		x_model = VectorXd(2);
		x_model(0) = x_sigma(0);
		x_model(1) = x_sigma(1);
		Z_list.col(idx) = x_model;
	}

	//cout << "Z_sig = \n" << Z_list << endl;

	// Calculate mean z from points
	VectorXd zpred = VectorXd(2);
	zpred << 0, 0;
	for ( int idx = 0; idx < 1 + 2 * n_aug_; idx++ )
	{
		zpred += weights_(idx) * Z_list.col(idx);
	}

	//cout << "Z_med = \n" << zpred << endl;

	// Calculate S
	MatrixXd S = MatrixXd(2, 2);
	S.setZero();
	for ( int idx = 0; idx < 1 + 2 * n_aug_; idx++ )
	{
		VectorXd z_diff = Z_list.col(idx) - zpred;
		S += weights_(idx) * z_diff * z_diff.transpose();
	}
	S(0, 0) += std_laspx_ * std_laspx_;
	S(1, 1) += std_laspy_ * std_laspy_;

	//cout << "S = \n" << S << endl;

	// Calculate T
	MatrixXd T = MatrixXd(5, 2);
	T.setZero();
	for ( int idx = 0; idx < 1 + 2 * n_aug_; idx++ )
	{
		VectorXd x_diff = Xsig_pred_.col(idx) - x_;

		while ( x_diff(3) > M_PI )
			x_diff(3) = x_diff(3) - 2 * M_PI;
		while ( x_diff(3) < -M_PI )
			x_diff(3) = x_diff(3) + 2 * M_PI;

		VectorXd z_diff = Z_list.col(idx) - zpred;
		T += weights_(idx) * x_diff * z_diff.transpose();
	}

	//cout << "T = \n" << T << endl;

	// actual measurement
	VectorXd zmeas = VectorXd(2);
	zmeas(0) = meas_package.raw_measurements_(0);
	zmeas(1) = meas_package.raw_measurements_(1);

	//cout << "Zmeas = \n" << zmeas << endl;

	VectorXd zinnov = VectorXd(2);
	zinnov = zmeas - zpred;

	//cout << "Zinnov = \n" << zinnov << endl;

	MatrixXd Sinv = S.inverse();

	MatrixXd K = T * Sinv;
	x_ = x_ + K * zinnov;

	//while ( x_(3) > M_PI )
	//	x_(3) = x_(3) - 2 * M_PI;
	//while ( x_(3) < -M_PI )
	//	x_(3) = x_(3) + 2 * M_PI;

	P_ = P_ - K * S * K.transpose();
	NIS_laser_ = zinnov.transpose() * Sinv * zinnov;

	//cout << "K = \n" << K << endl;
	//cout << "X = \n" << x_ << endl;
	//cout << "P = \n" << P_ << endl;
	cout << "e laser = \n" << NIS_laser_ <<endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
	/**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
	 */
	MatrixXd Z_list;
	Z_list = MatrixXd(3, 1 + 2 * n_aug_);

	// From all sigma points
	for( int idx = 0; idx < 1 + 2 * n_aug_; idx++ )
	{
		VectorXd x_sigma;
		x_sigma = VectorXd(5);
		x_sigma = Xsig_pred_.col(idx);

		VectorXd zpred_i = VectorXd(3);

		float px  = x_sigma(0);
		float py  = x_sigma(1);
		float v   = x_sigma(2);
		float yaw = x_sigma(3);

		if ( abs(px) < 1e-3 && abs(py) < 1e-3 )
		{
			px = 1e-3 * copysign(1.0, px);
			py = 1e-3 * copysign(1.0, py);
		}

		float rho  = sqrt(px * px + py * py);
		float phi  = atan2(py, px);
		float drho = (px * cos(yaw) * v + py * sin(yaw) * v) / rho;

		zpred_i(0) = rho;
		zpred_i(1) = phi;
		zpred_i(2) = drho;

		Z_list.col(idx) = zpred_i;
	}

	//cout << "Z_sig = \n" << Z_list << endl;

	// Calculate mean z from points
	VectorXd zpred = VectorXd(3);
	zpred << 0, 0, 0;
	for ( int idx = 0; idx < 1 + 2 * n_aug_; idx++ )
	{
		zpred += weights_(idx) * Z_list.col(idx);
	}

	//cout << "Z_med = \n" << zpred << endl;

	// Calculate S
	MatrixXd S = MatrixXd(3, 3);
	S.setZero();
	for ( int idx = 0; idx < 1 + 2 * n_aug_; idx++ )
	{
		VectorXd z_diff = Z_list.col(idx) - zpred;

		while ( z_diff(1) > M_PI )
			z_diff(1) = z_diff(1) - 2 * M_PI;
		while ( z_diff(1) < -M_PI )
			z_diff(1) = z_diff(1) + 2 * M_PI;

		S += weights_(idx) * z_diff * z_diff.transpose();
	}
	S(0, 0) += std_radr_   * std_radr_;
	S(1, 1) += std_radphi_ * std_radphi_;
	S(2, 2) += std_radrd_  * std_radrd_;

	//cout << "S = \n" << S << endl;

	// Calculate T
	MatrixXd T = MatrixXd(5, 3);
	T.setZero();
	for ( int idx = 0; idx < 1 + 2 * n_aug_; idx++ )
	{
		VectorXd x_diff = Xsig_pred_.col(idx) - x_;

		while ( x_diff(3) > M_PI )
			x_diff(3) = x_diff(3) - 2 * M_PI;
		while ( x_diff(3) < -M_PI )
			x_diff(3) = x_diff(3) + 2 * M_PI;

		VectorXd z_diff = Z_list.col(idx) - zpred;

		while ( z_diff(1) > M_PI )
			z_diff(1) = z_diff(1) - 2 * M_PI;
		while ( z_diff(1) < -M_PI )
			z_diff(1) = z_diff(1) + 2 * M_PI;

		T += weights_(idx) * x_diff * z_diff.transpose();
	}

	//cout << "T = \n" << T << endl;

	// actual measurement
	VectorXd zmeas = VectorXd(3);
	zmeas(0) = meas_package.raw_measurements_(0);
	zmeas(1) = meas_package.raw_measurements_(1);
	zmeas(2) = meas_package.raw_measurements_(2);

	//cout << "Zmeas = \n" << zmeas << endl;

	VectorXd zinnov = VectorXd(3);
	zinnov = zmeas - zpred;

	//cout << "Zinnov = \n" << zinnov << endl;

	MatrixXd Sinv = S.inverse();

	MatrixXd K = T * Sinv;
	x_ = x_ + K * zinnov;

	//while ( x_(3) > M_PI )
	//	x_(3) = x_(3) - 2 * M_PI;
	//while ( x_(3) < -M_PI )
	//	x_(3) = x_(3) + 2 * M_PI;

	P_ = P_ - K * S * K.transpose();
	NIS_radar_ = zinnov.transpose() * Sinv * zinnov;

	//cout << "K = \n" << K << endl;
	//cout << "X = \n" << x_ << endl;
	//cout << "P = \n" << P_ << endl;
	cout << "e radar = \n" << NIS_radar_ <<endl;
}
