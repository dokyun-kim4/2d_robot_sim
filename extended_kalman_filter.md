# Implementing an Extended Kalman Filter

## 0. Assignment Introduction

### 0.1 Welcome to Extended Kalman Filtering!

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

This file contains the walkthrough for the second part of the assignment. Before you start on this, make sure that your base simulator is working (walkthrough in `simulator.md`) and that you've completed the first part (walkthrough in `kalman_filter.md`)!

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

### 0.3 Resources for Understanding Extended Kalman Filters

Beyond the in-class lectures, here are a few additional resources you can use as alternate methods for understanding the Kalman Filter algorithm:
- Kalman Filter wikipedia: https://en.wikipedia.org/wiki/Extended_Kalman_filter
- Robert Labbe's Interactive Kalman Filtering Textbook (particularly chapter 11) https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
- Ivy's EKF Implementation: https://github.com/itannermahncke/extended-kalman-filtering

## 1. Nonlinearity In the Base Simulator

### 1.1 A Nonlinear Control Input

> File: `src/robot.py`

In the first part of this assignment, we defined a control input vector with a linear relationship to the state vector for which we'd like to make estimates:

$$x = [x, y, \theta]$$
$$u = [v_x, v_y, \omega]$$

We also considered another option for a control input vector, which had a nonlinear relationship to the state vector:

$$u = [v, \omega]$$

Since we'd like to explore the Extended Kalman Filter in this part of the assignment, we'll use this new control input vector to move our robot. The function that executes the given control schema is `robot_step_differential()`. You may have already implemented this in your base simulator, but if not, the guide for the function is in `simulator.md`. Essentially, `robot_step_differential()` should take in a linear velocity command and an angular velocity command as inputs, and output the expected change in state (dx, dy, and dtheta) given those commands.

**Implementation Action:** If you chose to implement `robot_step_differential()`, go ahead and skip this step. If not, go back and add it to your `src/robot.py` file!

> Coding tip: Reference code for this function is available on the `main` branch, if you'd like to get to the Extended Kalman Filter code more quickly.

### 1.3 Nonlinear Observations

> File: `src/sensors.py`

In the first part of this assignment, we defined a GPS sensor to measure our robot's position in the world. Now that we're implementing an EKF, we can integrate even more sensor information into our state estimate. This includes sensors whose observation spaces have a nonlinear relationship to the state space we're estimating!

Let's revisit the landmark pinger sensor, which measures the robot's range and bearing to all nearby landmarks. The observations from this sensor look like this (for each landmark):

$$z_i = [r_i, \phi_i]$$

These variables have a nonlinear relationship to our state space variables. Luckily, our EKF can handle that sort of thing! As a result, we'll be able to add information from this sensor into our state estimate.

**Implementation Action:** None in this step, as you should have already implemented the LandmarkPinger sensor in the base simulator!

Now we're ready to start writing our Kalman Filter!

## 2. Implementing the Kalman Filter's Prediction Step

> File: `src/extended_kalman_filter.py`

### 2.1 Initializing the Kalman Filter

First, let's set up a few key attributes. Just like the class `KalmanFilter`, the `ExtendedKalmanFilter` class's `__init__()` function takes in `dt` and `prior` as parameters. Go ahead and save both values to the proper class attributes before moving on.

**Implementation Action**: Set the two corresponding class attributes that correspond to the parameters `dt` and `prior`.

The process model for the EKF, `self.P`, is used in the exact same way as in linear Kalman Filtering. Recall that in a covariance matrix, the diagonal terms represent the variance (uncertainty) on each independent variable, while the non-diagonal terms indicate the joint variance of two variables. Go ahead and initialize `self.P` as a 3-by-3 identity matrix, just like we did in the first part.

**Implementation Action**: Initialize `self.P`, the process model.

> Coding tip: You can quickly construct the identity matrix using `np.eye()`.

Next, we need to define our state transition model. Until now, the implementation all been the same as the linear Kalman Filter. But because we are now using a new type of control input, the equations that calculate the EKF's future state prediction are nonlinear. As such, defining and using the state transition model will look a little different.

To do so, we are going to utilize [the Sympy library,](https://docs.sympy.org/latest/tutorials/intro-tutorial/matrices.html) which makes it simple to create and evaluate symbolic expressions. The library has been imported for you at the top of the file, as well as a set of symbols we'll be using (in particular, the state space and control space variables). You can reference these variables like any other variables, but they do not hold numerical data unless you explicitly substitute in values.

> Coding tip: To install Sympy on your computer, run the following in your terminal:
> `pip install sympy`

To start off, let's look at `self.f_xu`. This matrix will store the nonlinear equations used to transition from one state to the next. As you know from implementing `robot_step_differential()`, our equations of motion look like this:

$$
x_{t+1} = x_t + v_t*cos(\theta_t) *dt
$$

$$
y_{t+1} = y_t + v_t*sin(\theta_t) *dt
$$

$$
\theta_{t+1} = \theta_t + \omega_t *dt
$$

Go ahead and write these equations programatically using the symbolic variables that have been imported for you. Note that each row of the matrix corresponds to a different state variable!

**Implementation Action**: Initialize `self.f_xu`, the nonlinear state transition model, as a symbolic matrix. Use the provided equations of motion above, as well as the imported symbolic variables.

It's nice that we have this equation, but it's currently unusable in any of the standard Kalman Filter equations thanks to its nonlinearity. To linearize it, we need the [Jacobian matrix](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant). The Jacobian matrix is essentially a set of partial derivatives for a given multivariable function. It also represents a high-quality linear approximation of its source function at a specific point.

Knowing this, let's define `self.F`, the Jacobian of `self.f_xu` with respect to x, the state vector. We will do this by calling the [Sympy jacobian() function](https://docs.sympy.org/latest/modules/matrices/matrices.html#sympy.matrices.matrixbase.MatrixBase.jacobian) on `self.f_xu`, with the input to the function being a list of all variables in the state vector.

**Implementation Action**: Initialize `self.F`, the linearized state transition model, using `self.Fxu.jacobian()`.

> Coding tip: Set the input parameter as `Matrix([x, y, theta])`. This is a symbolic vector representing the state vector, which we linearize with respect to.

Now we have a symbolic matrix representing our linearized motion model! With that, we are ready to move on to the EKF prediction step.

### 2.2 The Prediction Step Equations

Let's refamiliarize ourselves with the equations of the prediction step for the linear Kalman Filter:

$$x_{t+1} = F * x_t + B * u_t$$

$$P = F * P * F^T + Q$$

The Extended Kalman Filter's prediction step looks like this:

$$F = \partial f(x,u)/\partial x$$

$$x_{t+1} = f(x,u)$$

$$P = F * P * F^T + Q$$

Note that we can still calculate our state vector using the nonlinear motion model just fine. It's only when we start doing linear algebra that we need to utilize the Jacobian!

You already have all of the pieces required for the prediction step in symbolic form. All that's left is to evaluate the symbolic matrices at our current state values and control input values. First, fill in each subtitution variable with its current value. Now we can do some evaluating!

To evaluate a symbolic matrix at a set of values, call the matrix's `subs()` function. The function expects one argument, which is a list of values to substitute. We already have this as the variable `subs`, so go ahead and pass that in! Next, take the result of `subs()` and pass it to the `sympy.matrix2numpy()` function, which simply converts the Sympy Matrix object to a numpy array. Now you have a numerical matrix to do math with!

**Implementation Action**: Inside the `predict()` function, set the value of each symbolic variable in `self.subs`. Then, follow the steps above to find a numerical matrix for `self.f_xu` and set that value in `fxu_eval`. Finally, repeat the steps for `self.F` and set that value in `F_eval`.

Now we have all the tools required to perform the prediction step mathematically. Go ahead and do that now!

**Implementation Action**: Inside the `predict()` function, calculate `self.x` and `self.P` using the numerical matrices you created.

> Coding tip: Don't forget to call `wrap_angle()` on your new state vector's heading variable!

> Coding tip: Like the linear Kalman Filter, a noise function `self.get_Q()` is provided for you. Make sure to call it each time you make a prediction for the process model!

### 2.3 Demonstrating The Prediction Step

> File: `src/main.py`

Now that we've put the EKF prediction step together, we can run our estimator along with the simulator and see what trajectory it predicts!

Since the linear Kalman Filter and the Extended Kalman Filter assume different control input vectors, we can't compare them directly. For now, alter your main file to initialize and run an instance of the `ExtendedKalmanFilter` rather than the linear `KalmanFilter`. You will also need to change your input file to be a list of linear and angular velocity commands, rather than a list of x, y, and angular velocity commands. When you're ready, run the simulator and visualize the output!

**Implementation Action**: Incorporate your Extended Kalman Filter into `main.py` and add logging/visualization peripherals for examining the utility of your filter.

You'll likely see that the Extended Kalman Filter's estimated trajectory is accumulating error over time, just like before. Without sensor updates of any kind, we are still performing dead reckoning and we are never correcting our estimates with any kind of observations. Let's move on to the update step and fix that!

## 3 Implementing The Kalman Filter's Update Step

> File: `src/extended_kalman_filter.py`

### 3.1 The Update Step Equations

Let's refamiliarize ourselves with key values of the update step:

$$S = H * P * H^T + R$$

$$K = P * H^T * S^{-1}$$

$$y = z - H * x$$

$$x_{t+1} = x_{t} + K * y$$

$$P_{t+1} = P - K * H * P$$

The Extended Kalman Filter's equations look slightly different due to the linearization step:

$$H = \partial h(x)/\partial x$$

$$S = H * P * H^T + R$$

$$K = P * H^T * S^{-1}$$

$$y = z - h(x)$$

$$x_{t+1} = x_{t} + K * y$$

$$P_{t+1} = P - K * H * P$$

Like the prediction step equations, the update step equations are nearly identical. However, h(x) (the measurement model) is now presumed to be nonlinear, and its linearized Jacobian matrix is used for all the linear algebra steps.

You'll notice that `update()` expects the following parameters:
- $H$: the linearized measurement model, a Jacobian matrix describing how the current predicted state looks in the observation space
- $R$: the measurement noise, AKA the covariance matrix associated with the observation

Also, the following two parameters are both optional. It is expected that the function will be passed one of these two values, but not both or neither:
- $z$: the observation, AKA the measurement taken by a sensor of the state variables (directly or indirectly)
- $y$: the residual, AKA the error between the measured observation and the expected observation given the predicted state

We set up the `update()` function in this way such that updates from linear and nonlinear observation spaces can both be processed. In the case of a linear observation space, like the GPS sensor's readings, the observation $z$ will be passed in directly. In the case of a nonlinear space, like the landmark pinger's readings, we will instead calculate $h(x)$, $H$, and $y$ inside of the sensor class and pass the values to this function.

Because we're outsourcing the calculation of the Jacobian and the residual to the nonlinear sensors, implementing the update step programatically will be identical to the linear Kalman Filter implementation. Go ahead and recreate the update equations programatically now!

**Implementation Action**: Implement the `update()` function.

> Coding tip: Don't forget to call `wrap_angle()` on your new state vector's heading variable!

### 3.2 Taking Measurements From The Simulator

> File: `src/sensors.py`

Now we'll want to hook up the update function to actual measurements taken by the robot's sensors in simulation. Each linear sensor will need to provide a measurement model, a noise model, and an observation. Each nonlinear sensor will need to provide a Jacobian linearized at the current state, a noise model, and a residual.

Let's take a look at the `LandmarkPinger` class. Just like in the predict step, we'll need to define our nonlinear measurement model `self.h_x` using Sympy. Recall the nonlinear relationship between our state space and our observation space:

$$r = \sqrt{(j - x)^2 + (k - y)^2}$$

$$\phi = tan^{-1}((j - x)/(k - y))$$

Note that $j$ and $k$ represent the x position and y position of the landmark we are observing, respectively.

Also, just like in the predict step, we'll want to utilize Sympy's `jacobian()` function to find `self.H`, the Jacobian matrix with respect to the state space, symbolically.

**Implementation action:** Initialize `self.h_x` as a symbolic matrix using the given equations for range and bearing. Initialize `self.H` as a symbolic matrix by calling `self.h_x.jacobian()` and using `Matrix([x, y, theta])` as an input.

Next, we need to fill in `self.H_eval()`, a function that evaluates the Jacobian at a given state for a given landmark. To implement this, you'll first need to extract the x and y position of a landmark by its ID (you have access to the environment's landmarks through the sensor's reference robot). Then, you'll want to save those values, alongside the values from the state parameter, to the substitution dictionary `self.subs`. Once you have `self.subs` properly set, you can use it to evaluate `self.H` as a numerical matrix!

**Implementation action:** Extract landmark x and y position using the given landmark ID and `self.robot.env.landmarks`, the list of known landmarks. Place the values into `self.subs` alongside values from the given state vector. Evaluate `H` at the values in `self.subs` and convert it to a numpy array before returning.

We also need to fill in `self.y()`, a function that calculates the residual between a given observation and the expected observation of a given landmark at a given state vector. To implement this, you'll again need to fill in the substitution dictionary `self.subs` using the given landmark's x and y position and the state vector. With those values, you can evaluate the nonlinear measurement model `self.h_x` and use it to calculate the residual!

**Implementation action:** Extract landmark x and y position using the given landmark ID and `self.robot.env.landmarks`, the list of known landmarks. Place the values into `self.subs` alongside values from the given state vector. Evaluate `H` at the values in `self.subs` and convert it to a numpy array. Finally, calculate the residual using the given EKF update equations and return it.

Finally, let's consider `R`, the measurement noise model. This is just like in the GPS sensor -- a diagonal matrix of the variance (or standard deviation, which we have, squared) for each observed variable. Recall that our range noise is proportional to distance, meaning the measurements get noisier when taken further away from a landmark. This means we'll have to recalculate R for each observation, rather than having a static property. This is implemented for you as the function `self.R()`, so give it a read and make sure you understand what it's doing!

Now you have all the pieces necessary to run the Extended Kalman Filter alongside your simulator!

### 3.3 Demonstrating The Full Extended Kalman Filter

> File: `src/main.py`

Just like the prediction step, we will replace `KalmanFilter.update()` in the main simulator loop with `ExtendedKalmanFilter.update()`. After `Robot.take_sensor_measurements()`, determine which exteroceptive sensors (if any) have returned observations, and perform an update for each. For GPS sensors, this should be essentially the same (just pass in `None` for the residual parameter).

The landmark pinger is a little more complicated, since it makes several observations. You will want to call `ExtendedKalmanFilter.update()` for every single landmark that returned a valid (within range, non-infinite) reading. When you call the `update()` function, you will actually be calling a `LandmarkPinger` function for every single input -- H, R, and y! Note that for this sensor, `None` should be passed in for the observation parameter.

Now you should be able to run the simulator with an Extended Kalman Filter estimating your robot's ground truth trajectory!

**Implementation Action**: Incorporate your Extended Kalman filter into `main.py` and adding logging/visualization peripherals for examining the utility of your filter.

## 4 Evaluating the EKF In Several Situations

Once everything is working, this is a great time to play around with the EKF. In particular, explore how the EKF's quality increases and decreases depending on availability of good sensor readings. For example, how does the EKF handle a complete absence of landmark observations? How quickly can it correct course when observing a landmark after a long absence? What happens if the robot crashes into a wall -- does the wheel encoder-based prediction throw it off for very long before measurements correct it? Finally, how does the EKF perform when you entirely remove landmark observations, GPS observations, or both?

Make sure to save a few plots that show the EKF's behavior in a few situations, and include your own thoughts on that behavior if you have time!
