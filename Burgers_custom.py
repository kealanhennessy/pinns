"""
@author: Kealan Hennessy <kealan@berkeley.edu>
@author: Maziar Raissi 
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from scaling_layer import ScalingLayer
import time

seed = 1234
np.random.seed(1234)
tf.random.set_seed(1234)

class PhysicsInformedNN:
    def __init__(self, X_train, u_train, num_layers, num_units, lb, ub, activation):
        
        # Boundary conditions
        self.lb = lb
        self.ub = ub

        # Activation function for surrogate DNN
        self.act = activation

        # Collect x, t (inputs) and u (labels) into tensors with uniform dtype
        self.x = tf.convert_to_tensor(X_train[:,0:1], dtype=tf.float32)
        self.t = tf.convert_to_tensor(X_train[:,1:2], dtype=tf.float32)
        self.u = tf.convert_to_tensor(u_train, dtype=tf.float32)
        
        self.num_layers = num_layers
        self.num_units = num_units

        # Use tf.Variable wrapper so lambda_1 and lambda_2 may act as learned parameters
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([-6.0], dtype=tf.float32)
        
        # Specify the vanilla Adam optimizer
        self.optimizer_Adam = tf.keras.optimizers.Adam()

        # Initialize weights and biases
        if self.act == 'relu':
            self.init_weights = tf.keras.initializers.HeNormal(seed=seed)
        else:
            self.init_weights = tf.keras.initializers.GlorotNormal(seed=seed)
        self.init_bias = tf.keras.initializers.Zeros()
        
        # Initialize the model and specify the optimization terrain
        self.surrogate_model = self.surrogate()
        self.var_list = self.surrogate_model.trainable_variables + [self.lambda_1, self.lambda_2]
    
    def surrogate(self):
        # Requires that we pass in concatenated x, t
        model = tf.keras.Sequential()
        model.add(ScalingLayer(lb=self.lb, ub=self.ub, input_shape=(2,)))
        for _ in range(self.num_layers - 1):
            model.add(tf.keras.layers.Dense(units=self.num_units, activation=self.act, 
                                            kernel_initializer=self.init_weights,
                                            bias_initializer=self.init_bias))
        model.add(tf.keras.layers.Dense(units=1, 
                                        kernel_initializer=self.init_weights,
                                        bias_initializer=self.init_bias, 
                                        activation=None))
        return model

    @tf.function
    def residual(self, x, t):
        # Compute the residual, or PINN
        lambda_1 = self.lambda_1
        lambda_2 = tf.exp(self.lambda_2)
        u = self.surrogate_model(tf.concat([x, t], 1))
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + (lambda_1 * u * u_x) - (lambda_2 * u_xx)
        return f
    
    def loss(self, u_pred, f_pred):
        # Compute MSE = MSE_u + MSE_f
        return tf.reduce_mean(tf.square(self.u - u_pred)) + tf.reduce_mean(tf.square(f_pred))

    def train(self, nIter_Adam=0, use_LBFGSB=False):
        start_time = time.time()
        for it in range(nIter_Adam + 1):
            with tf.GradientTape(persistent=True) as tape:
                u_pred = self.surrogate_model(tf.concat([self.x, self.t], 1))
                f_pred = self.residual(self.x, self.t)
                loss_value = self.loss(u_pred, f_pred)
            grads = tape.gradient(loss_value, self.var_list) # Compute gradients
            self.optimizer_Adam.apply_gradients(zip(grads, self.var_list)) # Apply gradients using the Adam optimizer
            if it % 500 == 0:
                elapsed = time.time() - start_time
                print('Adam iteration: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f, Time: %.2f' % 
                      (it, loss_value.numpy(), self.lambda_1.numpy(), np.exp(self.lambda_2), elapsed))
                start_time = time.time()

        if use_LBFGSB:
            opt_f = self.create_f()
            init_params = tf.dynamic_stitch(opt_f.idx, self.var_list)
            results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=opt_f, 
                                                    initial_position=init_params,
                                                    max_iterations=50000,
                                                    max_line_search_iterations=50,
                                                    num_correction_pairs=50,
                                                    f_absolute_tolerance=(1.0 * np.finfo(float).eps))
            opt_f.update(results.position)

    def predict(self, X):
        # Predict the values of u at spatiotemporal positions (x, t) 
        # using the learned lambda values stored in self.lambda_1, self.lambda_2
        x_in = tf.convert_to_tensor(X[:,0:1], dtype=tf.float32)
        t_in = tf.convert_to_tensor(X[:,1:2], dtype=tf.float32)
        u = self.surrogate_model(X)
        f = self.residual(x_in, t_in)
        return u, f
    
    def create_f(self):
        """
        @author: Kealan Hennessy
        @author: https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993
        """
        # Obtain the shapes of all trainable parameters in the model
        shapes = tf.shape_n(self.var_list)
        num_tensors = len(shapes)

        # Prepare requisite variables for later stitching and partitioning
        counter = 0
        idx = [] # stitch indices
        part = [] # partition indices
        for i, shape in enumerate(shapes):
            s = np.product(shape)
            idx.append(tf.reshape(tf.range(counter, counter + s, dtype=tf.int32), shape))
            part.extend([i] * s)
            counter += s
        part = tf.constant(part)

        @tf.function
        def update(one_dim_params):
            params = tf.dynamic_partition(one_dim_params, part, num_tensors)
            for i, (shape, param) in enumerate(zip(shapes, params)):
                self.var_list[i].assign(tf.reshape(param, shape))

        @tf.function
        def f(one_dim_params):
            with tf.GradientTape() as tape:
                update(one_dim_params) # Update model parameters via manual partitioning
                u_pred = self.surrogate_model(tf.concat([self.x, self.t], 1))
                f_pred = self.residual(self.x, self.t)
                loss = self.loss(u_pred, f_pred) # Calculate the loss

            # Calculate gradients and convert them to a 1-dimensional tf.Tensor
            grads = tape.gradient(loss, self.var_list)
            grads = tf.dynamic_stitch(idx, grads)

            f.iter.assign_add(1)
            if f.iter % 500 == 0:
                tf.print("L-BFGS-B iteration:", f.iter, "Loss:", loss, "Lambda_1: ", \
                         self.lambda_1[0], "Lambda_2: ", tf.exp(self.lambda_2[0]))

            # Store the loss value so we can retrieve it later
            tf.py_function(f.hist.append, inp=[loss], Tout=[])

            return loss, grads

        # Store function members for use outside scope
        f.idx = idx
        f.shapes = shapes
        f.part = part
        f.update = update
        f.iter = tf.Variable(0)
        f.hist = []

        return f