import tensorflow as tf

class ScalingLayer(tf.keras.layers.Layer):
    """
    @author: Kealan Hennessy
    Enforces boundary conditions when training surrogate network for PINN algorithm
    """
    def __init__(self, lb, ub, **kwargs):
        super(ScalingLayer, self).__init__(**kwargs)
        self.lb = lb
        self.ub = ub

    def build(self, input_shape):
        super(ScalingLayer, self).build(input_shape)

    def call(self, inputs):
        scaled_inputs = 2.0 * tf.divide((inputs - self.lb), (self.ub - self.lb)) - 1.0
        return scaled_inputs

    def compute_output_shape(self, input_shape):
        return input_shape