# Implementing a Linear Kalman Filter

## 0. Assignment Introduction

### 0.1 Welcome to Kalman Filtering!

This assignment has two goals: (1) help you practice and understand the fundamental math behind Kalman Filters, and (2) connect that math to a real-world state estimation scenario represented in our existing simulator. As you work through the assignment, keep both goals in mind, and try to reinforce both! If you find yourself struggling to understand what the math equations mean, or are struggling to translate those equations into code, there are resources listed below to support your learning.

The overall structure of this assignment follows:
* **Week 1: Linear (Regular) Kalman Filter**
    * Adding a linear control schema and GPS sensor to the base simulator
    * Implementing the Prediction Step
    * Implementing the Update Step
* **Week 2: Extended Kalman Filter**
    * Adding a nonlinear control schema to the base simulator
    * Implementing the Prediction Step
    * Implementing the Update Step
    * Implementation Comparisons

This file contains the walkthrough for the first two parts of the assignment. Before you begin, make sure your base simulator is working (walthrough in `simulator.md`). The walkthrough for the third part of the assignment, which covers Extended Kalman Filtering, can be found in `extended_kalman_filter.md`.

### 0.2 Kalman Filtering vs Extended Kalman Filtering

The Kalman Filter takes the following standard form:

$$x_k = F_kx_{k-1} + B_ku_k + w_k$$

$$z_k = H_kx_k + v_k$$

where $F_k$ is the state transition model capturing the transition from the previous state $x_{k-1}$ to the current state $x_k$; $B_k$ is the control-input model applied to the control vector $u_k$; w_k is _process_ noise with covariance $Q_k$; $z_k$ is a modeled measurement, $H_k$ is the observation model that maps true state space to observations, $v_k$ is the _observation_ noise with covariance $R_k$.


As we've discussed in class, Kalman Filters make two key assumptions. We are going to briefly discuss one of them, because it relates directly to our implementation: To use a Kalman Filter, the dynamics of the system we're estimating must be linear.

This can be tricky to understand at first. Essentially, the Kalman Filter algorithm relies heavily on linear algebra to generalize its computations. When we mathematically describe the relationship between different variables that affect each other, we'd like that relationship to be linear, so we can compute it using matrix math.

One example is the relationship between the state space and the control input (action) space. If our state vector tracks position, and our control input is velocity, this relationship is easily linearized with Euler's Method:

$$x_{t+1}=x_t + v_t*dt$$

This equation could be simply slotted into the state transition matrix (F) and used to calculate predictions. However, if the relationship between state space and control space is not linear, we will have to linearize it to take advantage of the Kalman Filter's efficient algorithm.

Another example is the relationship between the state space and the observation space. A GPS sensor might observe the robot's position, which naturally has a linear relationship with the state vector since it tracks the same value. That relationship is easily slotted into the measurement model matrix (H). However, a camera that determines the robot's position relative to a known landmark is observing variables (specifically, bearing and range) that have a nonlinear relationship with the robot's position:

$$r = \sqrt{(x - l_x)^2 + (y - l_y)^2}$$ 

$$\phi = tan^{-1}([y - l_y] / [x - l_x]) - \theta$$

In this case, H would need to be linearized.

Anything involving linearization becomes an Extended Kalman Filter (EKF).

> For the first half of the assignment, which we recommend spending one week on, the walkthrough will focus on developing a regular Kalman Filter for our simulator. The second half of the assignment will focus on adding linearization to the estimator, making it an Extended Kalman Filter.

### 0.3 Resources for Understanding Kalman Filters

Beyond the in-class lectures, here are a few additional resources you can use as alternate methods for understanding the Kalman Filter algorithm:
- Kalman Filter wikipedia: https://en.wikipedia.org/wiki/Kalman_filter
- CompRobo Kalman Filtering Recitation: https://github.com/comprobo25/recitation_examples/blob/main/kalman_filters/
- Robert Labbe's Interactive Kalman Filtering Textbook (particularly chapters 6 and 7) https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
- Ivy's EKF Implementation: https://github.com/itannermahncke/extended-kalman-filtering

## 1. Linearity In the Base Simulator

### 1.1 A Linear Control Input

> File: `src/robot.py`

We briefly discussed linearity in some toy examples, but let's consider the scenario that we actually built in our simulator. First, here is our state vector:

$$x = [x, y, \theta]$$

We'll need to determine if our state space has a linear relationship to our control input space. This depends on whether or not you chose to implement `robot_step_translational()` or `robot_step_differential()`. For `robot_step_translational()`, our control input vector looks like this:

$$u = [v_x, v_y, \omega]$$

Since the control input vector only contains derivatives of state vectors, this relationship is linear by Euler's Method! However, the second method, `robot_step_differential()` is a different story:

$$u = [v, \omega]$$

This control vector has variables with a nonlinear relationship to the state vector's variables. As such, if we were trying to estimate the state of a robot with this type of drive, we would need an Extended Kalman Filter.

**Implementation Action:** If you chose to implement `robot_step_translational()`, go ahead and skip this step. If not, go back and add it to your `src/robot.py` file! Either way, don't delete your `robot_step_differential()` function if you have it -- we'll use that mode for the Extended Kalman Filter. Make sure that your `WheelEncoder` sensor class matches the translational mode as well.

> Coding tip: Reference code for this function is available on the `main` branch, if you'd like to get to the Kalman Filter code more quickly.

### 1.3 Linear Observations

> File: `src/sensors.py`

We also need to examine the relationship between our state space and our measurement space. Currently, the only exteroceptive sensor we have is the landmark pinger (remember that our wheel encoders are supplying our control input info for the prediction step, not observations for the update step!). Landmark pinger observations look like this:

$$z = [r, \phi]$$

Like we showed before, these variables have a nonlinear relationship to our state space variables. When we implement the Extended Kalman Filter, we'll use these measurements, but for now, we are going to add in a new sensor whose observations are easier to linearly relate to our state space.

**Implementation Action:** Let's add a GPS sensor to `src/sensors.py`. Like the other sensor classes, your new `GPS` class should have the following features:
- Inherits from `SensorInterface`
- `__init__()` function with a name, reference robot, and interval
- Noise constants for each observed state (x position and y position)
- `sample()` function that applies noise to ground truth x position and y position accessed in the robot's `env` property and returns the values

> Coding tip: Reference code for this function is available on the `main` branch, if you'd like to get to the Kalman Filter code more quickly.

> Coding tip: Make sure to also add an instance of this class to your `Robot` class and utilize the `GPS.sample()` function in `Robot.take_sensor_measurements()`.

Now that we've confirmed that our system is linear in both control and observations, we're ready to start writing our Kalman Filter!

## 2. Implementing the Kalman Filter's Prediction Step

> File: `src/kalman_filter.py`

### 2.1 Initializing the Kalman Filter

First, let's set up a few key attributes. Notice that the `KalmanFilter` class's `__init__()` function takes in two parameters: `dt`, which you'll recognize as an attribute shared by the `Environment` class; and `prior`, which describes an initial estimate of the robot's state vector. Make sure to save both of these as class attributes before moving on.

Next, let's talk about `self.P`, the process model. This covariance matrix describes the uncertainty present in our estimates of each state variable. Recall that in a covariance matrix, the diagonal terms represent the variance (uncertainty) on each independent variable, while the non-diagonal terms indicate the joint variance of two variables. We have to choose a prior value for this matrix. A classic choice is simply the identity matrix, but you can play with the scale and even add non-diagonal terms if you'd like to explore further!

> Coding tip: You can quickly construct the identity matrix using `np.eye()`.

Now that we're modeling our state and state uncertainty properly, we'll need to model how our state changes from timestep to timestep. This information is encoded in two matrices: the state transition model `self.F`, which describes how the state changes with *no* inputs to the system; and the control input model `self.B`, which describes how the state changes in response to control inputs.

`self.F` is simple in this case, because all of our state variables are independent (you can see a more interesting case [here, under 'Tracking A Dog'](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/06-Multivariate-Kalman-Filters.ipynb).) As a result, the state transition matrix is simply an identity matrix. Physically, this means that with no control inputs present, no state variables will change!

`self.B` is more interesting. One way to think about this matrix is that it should convert the control input vector to a vector describing the change in value for each state variable:

$$[\Delta x,\Delta y, \Delta \theta] = B * [v_x, v_y, \omega]$$

In practice, this will look like a diagonal matrix scaled by the unit timestep `dt`. This strategy employs Euler's Method to approximate small changes in each state variable using the value of control input variables when they represent derivatives of those state variables.

**Implementation Action**: Fill in the initialization function for the Kalman filter class for attributes `self.F`, `self.P`, and `self.B`.

> Coding tip: There is a new function `wrap_angle()` in `src/utils.py`. You should use this each time you update the robot's heading, to prevent accumulating non-normalized angle values that might cause calculation errors later on.

### 2.2 The Prediction Step Equations

Let's refamiliarize ourselves with the equations of the prediction step:

$$x_{t+1} = F * x_t + B * u_t$$

$$P = F * P * F^T + Q$$

The final attribute to initialize is Q, the process noise, which is essentially a catch-all noise model to acknowledge that the world doesn't operate as perfectly as the state transition might make it seem. Wind, wheel slippage, and other unmeasurable disturbances are all represented here. The simplest way to initialize Q is as a white noise/Gaussian distribution. We have done this for you in the `get_Q()` function. Take some time to try out different values for the standard deviation parameter and see how that affects the noise!

Tuning the Q parameter is a finnicky part of Kalman Filtering -- too little noise, and your filter is blindsided by natural disturbances in the world. Too much noise, and your filter is unable to parse out good estimates. One alternate method of defining Q is by basing it on known noise in your velocity commands. To learn more about this technique, check out [Chapter 11 of the RLabbe Kalman Filtering textbook](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/11-Extended-Kalman-Filters.ipynb)!

Now that we've set up all of our matrices, this is just a matter of recreating the equations in the `predict()` function. Go ahead and do this now!

**Implementation Action**: Fill in the `self.Q` attribute in the initialization function, and implement the `predict()` function.

### 2.3 Demonstrating The Prediction Step

> File: `src/main.py`

Last week, you got your simulator to the point of generating ground truth data and noisy sensor data, and you hopefully created some visualization code to plot that data after the fact (if not, this is a good time to go back and add some visualizer code!). Now that we've put the prediction step together, we can run our estimator along with the simulator and see what trajectory it predicts!

To do this, navigate to your main file and initialize a new instance of the `KalmanFilter` class with parameters that match whatever you passed your `Environment` class. Then, in your main simulator loop, you'll want to call your new `KalmanFilter.predict()` function immediately after calling `Robot.take_sensor_measurements()`. In our case, the control input vector comes from the robot's wheel encoder readings. Next, save the resultant state estimates at each timestep and plot them alongside your plot of the ground truth trajectory to see how your Kalman Filter is performing!

**Implementation Action**: Incorporate your Kalman filter into `main.py` and adding logging/visualization peripherals for examining the utility of your filter.

You'll likely see that the Kalman Filter's estimated trajectory isn't perfect, and is likely even accumulating error over time. This is because we're effectively just doing dead reckoning right now -- using proprioceptive sensor knowledge only to figure out where the robot is. With no corrective updates from exteroceptive sensors, the filter has no way of reducing its uncertainty. This is why we need the update step.

## 3 Implementing The Kalman Filter's Update Step

> File: `src/kalman_filter.py`

### 3.1 The Update Step Equations

Let's refamiliarize ourselves with key values of the update step:

$$S = H * P * H^T + R$$ 

Represents the total uncertainty, where the process model P is propagated through the measurement model H to put it in the same units as the measurement noise R.

$$K = P * H^T * S^{-1}$$ 

Is the Kalman Gain, AKA percentage of total uncertainty that is from the estimator, not the measurement.

$$y = z - H * x$$ 

Is the residual, AKA error between observation and expected observation given current state estimate.

And here are the equations of the upate step:

$$x_{t+1} = x_{t} + K * y$$

$$P_{t+1} = P - K * H * P$$

You'll notice that `update()` already expects the following parameters:
- $z$: an observation, AKA a measurement taken by an exteroceptive sensor
- $H$: the measurement model, describing the mathematical relationship between the state space and the sensor's measurement space
- $R$: the measurement noise, AKA the covariance matrix associated with the observation

**Implementation Action**: Implement the `update()` function.

### 3.2 Taking Measurements From The Simulator

> File: `src/sensors.py`

Just like with the predict step, we'll want to hook up this function to actual measurements taken by the robot's sensors in simulation. While the z parameter comes from taking samples, which the simulator can already do, we need to define the measurement model H and noise model R for any exteroceptive sensor informing our updates. Right now, this only includes our GPS sensor, but if you create other sensors in the future, keep this in mind!

Let's head over to the GPS sensor and add two new attributes in its `__init__()` function: H and R. R is simply a diagonal matrix containing the variance for each variable observed by the sensor. Note that our noise parameters represent standard deviation values -- you need to square them to calculate the variance!

The measurement model describes which state variables are directly observed by a sensor, and which are not. Mathematically, it is a matrix with one row per measurement variable and one column per state variable. In the case of the GPS sensor, there are two measurement variables (x position, y position) and three state variables (x position, y position, heading). Each row should contain a 1 in the space of the state variable corresponding to the row's measurement variable, and a zero otherwise.

**Implementation Action**: Add attributes `self.H` and `self.R` to the GPS sensor class.

Now you have all the pieces necessary to run the Kalman Filter alongside your simulator!

### 3.3 Demonstrating The Full Kalman Filter

> File: `src/main.py`

Just like the prediction step, we will call `KalmanFilter.update()` in the main simulator loop. After `Robot.take_sensor_measurements()`, determine which exteroceptive sensors (if any) have returned observations, and perform an update for each. In the current case, this means calling `KalmanFilter.update()` each time a new GPS observation arrives. Also, be sure to log the Kalman Filter's estimated state vector after the update step instead of after the prediction step!

Now when you plot the results of the filter compared to the ground truth, you should see that the Kalman Filter's estimated trajectory is much closer to reality! Generally speaking, in Kalman Filtering, uncertainty grows in the prediction step and shrinks during the update step. Following this, it makes sense that our unbounded error accumulation has been corrected now that we are encorporating updates.

**Implementation Action**: Incorporate your Kalman filter into `main.py` and adding logging/visualization peripherals for examining the utility of your filter.

## 4 Looking Ahead To The Extended Kalman Filter

This concludes the implementation of a basic Kalman Filter. In the second half of this assignment, we will modify this code to implement the Extended Kalman Filter. We'll modify our control inputs and add in updates from the landmark pinger sensor, both of which will require linearization. Then, we'll compare the results of the EKF to the results from the basic Kalman Filter!
