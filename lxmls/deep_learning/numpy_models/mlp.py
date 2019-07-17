import numpy as np
from lxmls.deep_learning.mlp import MLP
from lxmls.deep_learning.utils import index2onehot, logsumexp


class NumpyMLP(MLP):
    """
    Basic MLP with forward-pass and gradient computation in Numpy
    """

    def __init__(self, **config):

        # This will initialize
        # self.config
        # self.parameters
        MLP.__init__(self, **config)

    def predict(self, input=None):
        """
        Predict model outputs given input
        """
        log_class_probabilities, _ = self.log_forward(input)
        return np.argmax(np.exp(log_class_probabilities), axis=1)

    def update(self, input=None, output=None):
        """
        Update model parameters given batch of data
        """

        gradients = self.backpropagation(input, output)

        learning_rate = self.config['learning_rate']
        num_parameters = len(self.parameters)
        for m in np.arange(num_parameters):

            # Update weight
            self.parameters[m][0] -= learning_rate * gradients[m][0]

            # Update bias
            self.parameters[m][1] -= learning_rate * gradients[m][1]

    def log_forward(self, input):
        """Forward pass for sigmoid hidden layers and output softmax"""

        # Input
        tilde_z = input
        layer_inputs = []

        # Hidden layers
        num_hidden_layers = len(self.parameters) - 1
        for n in range(num_hidden_layers):

            # Store input to this layer (needed for backpropagation)
            layer_inputs.append(tilde_z)

            # Linear transformation
            weight, bias = self.parameters[n]
            z = np.dot(tilde_z, weight.T) + bias

            # Non-linear transformation (sigmoid)
            tilde_z = 1.0 / (1 + np.exp(-z))

        # Store input to this layer (needed for backpropagation)
        layer_inputs.append(tilde_z)

        # Output linear transformation
        weight, bias = self.parameters[num_hidden_layers]
        z = np.dot(tilde_z, weight.T) + bias

        # Softmax is computed in log-domain to prevent underflow/overflow
        log_tilde_z = z - logsumexp(z, axis=1, keepdims=True)

        return log_tilde_z, layer_inputs

    def cross_entropy_loss(self, input, output):
        """Cross entropy loss"""
        num_examples = input.shape[0]
        log_probability, _ = self.log_forward(input)
        return -log_probability[range(num_examples), output].mean()

    def backpropagation(self, input, output):
        """Gradients for sigmoid hidden layers and output softmax"""

        # Run forward and store activations for each layer
        log_prob_y, layer_inputs = self.log_forward(input)
        prob_y = np.exp(log_prob_y)

        num_examples, num_clases = prob_y.shape
        num_hidden_layers = len(self.parameters) - 1

        # For each layer in reverse store the backpropagated error, then compute
        # the gradients from the errors and the layer inputs
        errors = []

        # ----------
        # Solution to Exercise 2

        # error at softmax layer
        I = index2onehot(output, num_clases)
        non_linear_error = (prob_y - I) / num_examples
        errors.append(non_linear_error)

        for n in reversed(range(num_hidden_layers)):

            # backpropagation through linear layer
            weights_n, bias_n = self.parameters[n+1]
            linear_error = np.dot(errors[-1], weights_n)

            # backpropagation through sigmoid
            non_linear_error = linear_error * layer_inputs[n+1] * (1 - layer_inputs[n+1])
            errors.append(non_linear_error)

        # update weights
        gradients = []
        errors.reverse()
        for n in range(len(self.parameters)):
            weights, bias = self.parameters[n]
            error = errors[n]
            layer_input = layer_inputs[n]

            gradient_weight = np.zeros(weights.shape)
            for l in range(num_examples):
                # error[l, :] --> previous error
                # input[l, :] --> output of previous layer, a.k.a. the input of this layer
                aux = np.outer(error[l, :], layer_input[l, :])
                gradient_weight += aux

            # Bias gradient
            gradient_bias = np.sum(error, axis=0, keepdims=True)

            gradients.append((gradient_weight, gradient_bias))

            # another function updates the weights and the bias, so we do NOT need to run this here!
            # # SGD update
            # learning_rate = self.config['learning_rate']
            # weights = weights - learning_rate * gradient_weight
            # bias = bias - learning_rate * gradient_bias
            #
            # # update parameters
            # self.parameters[n][0] = weights
            # self.parameters[n][1] = bias

        # End of solution to Exercise 2
        # ----------

        return gradients
