# KalmanF
All kinds of Kalman Filter with Python and C++


### Kalman Filter Basics

* x_k = A * x_k-1 + B * u_k + w_k : Process equation, Dynamic equation

* z_k = H * x_k + v_k : Measurement equation

    * x_k : system state at k
    * x_k-1 : system state at k-1
    * z_k : measurement at k
    * u_k : external control at k

* w_k : process noise, p(w) ~ N(0, Q)
* v_k : measurement noise, p(v) ~ N(0, R)
* A : state transition model, n x n matrix
* B : optional control - input model, n x l matrix
* H : observation model, m x n matrix
* Q : process noise covariance matrix
* R : measurement covariance matrix

* init - predict - correct
    * init
        * initial estimation, model parameter
    * predict (statePre)
        * x'_k = A * x_k-1 + B * u_k
        * P'_k = A * P_k-1 * A^T + Q
    * update (statePost)
        * K_t = P'_k * H^T * ( H * P'_K * H^T + R )^-1
        * x_k = x'_k + K_t * (z_k - H * x'_k)
        * P_k = (I - K_k * H ) * P'_k






