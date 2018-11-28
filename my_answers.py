#  Student: Eric Olsen
#  There are many self-apparent notes below, these mostly are in place to assist me
#  in future projects when I need to review previous work. These would likely be
#  pretty self-apparent facts to a professional developer.

import numpy as np

# these class fields are all defined at the bottom of the program
class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        # We randomly set these to 1/square-root of the total so it's closer to zero
        # Introducing a bias to lower numbers assists because we're using sigmoid activation functions
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        
        self.lr = learning_rate# parameter from the bottom
        
        self.activation_function = lambda x : 1 / (1 + np.exp(-x)) # sigmoid activation function
        

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
                      - After one-hot encoding is complete, this is 59 colums per entry
                      - Many equate to time, but there are 4 other significant and normalized factors:
                          - casual vs. registered (which sums to cnt)
                          - temperature
                          - humidity
                          - windspeed
            targets: 1D array of target values
                      - We are measuring as targest the cnt which sum from the registered and 
                      casualusers
        
        '''
        n_records = features.shape[0] #13,347 records for this particular data set
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # forward pass defined below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, 
                                                                        X, y, 
                                                                        delta_weights_i_h, 
                                                                        delta_weights_h_o)
        # defined below    
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        # This should just be 
            #1) a dot product of weights and inputs, 
            #2) the activation function on the resulting vectors
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # 1
        hidden_outputs = self.activation_function(hidden_inputs) # 2

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        
        # Lesson Learned:
        # I think I was tripping myself up here by doing a second sigmoid (as the courses did)
        # If I just assign the final outputs from the final inputs, this is my answer.
        # This became a significant issue with my back propogations.
        final_outputs = final_inputs
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        # This becamse pretty cluttered with comments. There are few comments below so the equations 
        # have emphasis

        error = y - final_outputs
        
        # Lesson learned: Order DOES matter in numpy when using the dot product in this manner
        hidden_error = np.dot(self.weights_hidden_to_output, error)
        
        # Since there is no second activation function, this is just an assignment operation 
        output_error_term = error
        
        # y - y_hat * f'(W a)
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        
        # ensure that we multiply by a column-based and not row-based vector
        delta_weights_i_h += hidden_error_term * X[:,None]
        
        # again, ensure we multiply a column and not a row-based vector
        delta_weights_h_o += output_error_term * hidden_outputs[:,None]
        
        return delta_weights_i_h, delta_weights_h_o

    
    
    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        # use the average for weigh updates calculated by a division of the number of records
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records 
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    
    # I belive this function mostly serves to print output from the calling program.
    # Under the hood the .train function is also implementing these function calls.
    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        #final_outputs = self.activation_function(final_inputs) # signals from final output layer 
        final_outputs = final_inputs
    
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 3500 
learning_rate = 0.5

hidden_nodes = 25
output_nodes = 1
