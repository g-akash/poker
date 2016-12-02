This is an implementation of neural network from scratch for predicting poker hand. The train and test data is present in the folder. It takes about 30mins to train.


The mechanism is simple. I did some feature engineering which basically means represented the cards in terms of features. card with number n and type t is represented as a vector of 13 + 4 entries where nth and 13+t entry is 1 and rest are 0. Also output has 10 values where ith value represent that the set has i as output. We use converter function this conversion from given input formats and also for converting back from our output format to actual output format.

The rest of the mechanism is simple. We simulate the neural network through matrix multiplication. where on each level we multiply the input vector by weight matrix to get the sum vector of next level. Then We apply the sigmoid function on this vector elements to get inputs for the next level. Rest we had some trial and error to optimize and get good results.