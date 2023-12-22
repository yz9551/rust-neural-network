pub mod matrix;

use matrix::{Matrix, Vector};
use std::fs;
use std::io::prelude::Read;
use serde::{Serialize, Deserialize};

type Data=f32;

pub trait Layer {
    fn forward_propagate(&self, inputs: &Vector) -> Vector;
    fn backward_propagate(&mut self, inputs: &Vector, output_error: &Vector, learning_rate: Data) -> Vector;
}

#[derive(Serialize, Deserialize)]
pub struct Dense {
    weights: Matrix,
    biases: Vector,
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize) -> Box<dyn Layer>{
        let weights = Matrix::new(output_size, input_size);
        let biases = Vector::new(output_size);
        Box::new(Dense { weights, biases })
    }
}

impl Layer for Dense {
    fn forward_propagate(&self, inputs:  &Vector) -> Vector {
        &(&self.weights * inputs) + &self.biases
    }

    fn backward_propagate(&mut self, inputs: &Vector, output_error: &Vector, learning_rate: Data) -> Vector {
        let input_error = (&self.weights).transpose() * output_error;
        self.weights -= output_error * inputs.transpose() * learning_rate;
        self.biases -= &(output_error * learning_rate);
        input_error

    }
}

//#[derive(Serialize, Deserialize)]
pub struct Activation {
    activation_function: fn(&Data) -> Data,
    function_derivative: fn(&Data) -> Data,
}

impl Activation {
    pub fn new (activation_function: fn(&Data) -> Data, function_derivative: fn(&Data) -> Data) -> Box<dyn Layer> {
        Box::new( Activation { activation_function, function_derivative } )
    }
}

impl Layer for Activation {
    fn forward_propagate(&self, inputs: &Vector) -> Vector {
        Vector::from_data(inputs.iter().map(self.activation_function).collect())
    }
    fn backward_propagate(&mut self, inputs: &Vector, output_error: &Vector, _learning_rate: Data) -> Vector {
        //assert_eq!(inputs.size(), output_error.size());
        Vector::from_data(
            (0..inputs.size())
                .map(|i|
                    (self.function_derivative)(&inputs[i]) * output_error[0])
                .collect()
        )
    }
}

//#[derive(Serialize, Deserialize)]
pub struct Network {
    layers: Vec<Box<dyn Layer>>,
    inputs: Vec<Vector>,
    loss: fn(&Vector, &Vector) -> Data,
    loss_prime: fn(&Vector, &Vector) -> Vector,
}

impl Network {
    pub fn new(loss: fn(&Vector, &Vector) -> Data, loss_prime: fn(&Vector, &Vector) -> Vector) -> Self {
        Self { layers:vec![], inputs: vec![], loss, loss_prime}
    }

    pub fn add(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer)
    }

    pub fn forward_propagate(&mut self, input: Vector) -> Vector {
        self.inputs.push(input);
        for layer in self.layers.iter() {
            //println!("input: {:?}", &self.inputs[self.inputs.len()-1]);
            self.inputs.push(layer.forward_propagate(&self.inputs[self.inputs.len()-1]));
        }
        self.inputs.pop().unwrap()
    }

    fn backward_propagate(&mut self, mut output_error: Vector, learning_rate: Data) {
        for layer in self.layers.iter_mut().rev() {
            //println!("input: {:?}", &self.inputs[self.inputs.len()-1]);
            output_error = layer.backward_propagate(&self.inputs.pop().unwrap(), &output_error, learning_rate);
        }
    }

    pub fn train(&mut self, input_data: Vec<Vector>, output_data: Vec<Vector>, learning_rate: Data, epochs: usize) {
        println!("Training with learning_rate: {} for {} epoch(s)", learning_rate, epochs);
        let sample_size = input_data.len();
        assert_eq!(sample_size, output_data.len());
        for epoch in 0..epochs {
            let mut error: Data = 0.0;
            for i in 0..sample_size {
                let output = self.forward_propagate(input_data[i].clone());
                error += (self.loss)(&output, &output_data[i]);
                //println!("backward propagation");
                self.backward_propagate((self.loss_prime)(&output, &output_data[i]), learning_rate);
            }
            if epoch % 10 == 0 {
                println!("epoch: {}, loss: {}", epoch, error);
            }
        }
    }

    pub fn load(&mut self, filename: &str) {
        let mut savefile = fs::File::open(filename).expect("file open failed");
        let mut contents = String::new();
        savefile.read_to_string(&mut contents).expect("read from file failed");
        //self = ron::from_str(&contents).expect("deserialization failed");
    }

    pub fn save(&mut self, filename: &str) {
        let savefile = fs::File::open(filename).expect("file open failed");
        //savefile.write_all(ron::to_string(self).expect("serialization failed")).expect("write to file failed");
    }

}


