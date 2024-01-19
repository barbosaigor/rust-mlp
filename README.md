# Multilayer perceptron in rust
This is a **just-for-fun** project I was curious about how to implement an MLP from scratch in Rust then I gave it a shot.  

Interesting readings: 
- [Using neural nets to recognize handwritten digits](http://neuralnetworksanddeeplearning.com/chap1.html)
- [How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)

### Building your neural network
```rust
let (input, input_labels) = read("./train.csv");
// The first inner layer has 10 neurons, with ReLu activation function
let layer1 = perceptron::PerceptronLayer::new(10, input.nrows(), activation::relu);
// The output layer has 10 neurons, with softmax activation function
let layer2 = perceptron::PerceptronLayer::new(10, layer1.weights.nrows(), activation::soft_max);

// You have to push them in the MLP model
let mlp = MLP::new().push(layer1).push(layer2);

const ALPHA: f64 = 0.3;
const EPOCHS: u32 = 150;
let mut trainer: Trainer = Trainer::new(mlp, ALPHA, EPOCHS);
trainer.train(&input, &input_labels);

// do predictions
```