import tensorflow as tf

__all__ = ['RMSProp']

class RMSProp(tf.train.Optimizer):
    def __init__(self, learning_rate, var_list, beta=0.999, epsilon=1e-8, name='rmsprop'):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.variables = var_list
        self.name = name
        self._init()

        super(RMSProp, self).__init__(learning_rate, name=name)

    def apply_gradients(self, steps_and_vars, *args, **kwargs):
        return super(RMSProp, self).apply_gradients(step_and_vars, *args, **kwargs)

    def preconditioning(self, grads_and_vars):
        covs_and_vars = self._update_cov(grads_and_vars, self.beta)
        beta_update_op = self._beta_power.assign(self._beta_power * self.beta)
        with tf.control_dependencies([beta_update_op]):
            return self._compute_steps(grads_and_vars, covs_and_vars)

    def precon_wo_update_cov(self, grads_and_vars):
        covs_and_vars = [(self._zeros_slot(var, "covariance", self.name), var) for _, var in grads_and_vars]
        return self._compute_steps(grads_and_vars, covs_and_vars)

    def _compute_steps(self, grads_and_vars, covs_and_vars):
        steps = []
        for (grads, var1), (cov, var2) in zip(grads_and_vars, covs_and_vars):
            if var1 is not var2:
                raise ValueError("The variables referenced by the two arguments "
                                 "must match.")

            cov = cov / (1 - self._beta_power)
            step = grads / (tf.sqrt(cov) + self.epsilon)
            steps.append(step)
        return steps

    def _init(self):
        first_var = min(self.variables, key=lambda x: x.name)
        with tf.colocate_with(first_var):
            self._beta_power = tf.Variable(1.0, name="beta_power", trainable=False)

    def _update_cov(self, vecs_and_vars, decay):
        def _update_covariance(vec, var):
            covariance = self._zeros_slot(var, "covariance", self.name)
            with tf.colocate_with(covariance):
                # Compute the new velocity for this variable.
                new_covariance = decay * covariance + (1 - decay) * vec ** 2

                # Save the updated velocity.
                return (tf.identity(covariance.assign(new_covariance)), var)

        # Go through variable and update its associated part of the velocity vector.
        return [_update_covariance(vec, var) for vec, var in vecs_and_vars]
