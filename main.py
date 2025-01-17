import numpy
#for plotting arrays
import matplotlib.pyplot as mathplot
#for sigmoid activation function
import scipy.special as scipyspecial

class neuralNetwork:
    #initializing the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #link weight matrices
        #using the normal distribution to keep the variance constant
        self.wih = numpy.random.normal(0.0,pow(self.inodes, -0.5), (self.hnodes, self.inodes))  #weight from input to hidden
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))  #weight from hidden to output

        #learning rate
        self.lr = learningrate

        #activation function is the sigmoid function 1/(1+e^-x)
        self.activation_function = lambda x: scipyspecial.expit(x)

        pass

    #train the neural network
    def train(self, inputs_list, targets_list):
        #convert inputs list to 2d array and transpose to be a column vector
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)

        #calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signal into final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        #calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        #error is the (target - actual)
        output_errors = targets - final_outputs

        #hidden layer error is the output_errors, split proportionaly to thier weight
        #and recombined at the hidden node
        hidden_errors = numpy.dot(self.who.T, output_errors)

        #update the weights for the link between the hidden and output layers 
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        #update the weights for the link between the input and hidden layer
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass

    #query the neural network
    def query(self, inputs_list):
        #convert inputs list to 2d array and transpose to have input as column vectors
        inputs = numpy.array(inputs_list, ndmin=2).T

        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)

        #calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signal into final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        #calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs    

#number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

#learning rate
learning_rate = 0.3

#initialize neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#load the mnist training data
data_file = open("../mnist_dataset/mnist_train_100.csv", 'r')
data_list = data_file.readlines()
data_file.close()

#taining the neural network
for record in data_list:
    all_values = record.split(',')
    inputs = (numpy.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)



#load the mnist test data
data_file = open("../mnist_dataset/mnist_test_10.csv", 'r')
data_list = data_file.readlines()
data_file.close()

#test the neural network
all_values = data_list[0].split(',')
print(all_values[0])
# # Display the image
image_array = numpy.asarray(all_values[1:], dtype=float).reshape((28, 28))
mathplot.imshow(image_array, cmap='Greys', interpolation='None')
mathplot.show()

result = n.query((numpy.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01)
print(result)









