use neural_network::{Network, Dense, Activation};
use neural_network::matrix::Vector;
use std::{io, iter};

fn relu(a: &f32) -> f32 { if *a > 0.0 { *a } else { 0.0 } }
fn relu_prime(a: &f32) -> f32 { if *a > 0.0 { 1.0 } else { 0.0 } }
fn mse(output: &Vector, actual: &Vector) -> f32 {
    iter::zip(output.iter(), actual.iter())
        .map(|(a, b)| (a - b) * ( a - b))
        .sum::<f32>() / output.len() as f32
}
fn mse_prime(output: &Vector, actual: &Vector) -> Vector {
    iter::zip(output.iter(), actual.iter())
        .map(|(a, b)| 2.0 * (a - b) / output.len() as f32)
        .collect::<Vec<_>>()
        .into()
}

fn main() {
    let mut my_net = Network::new(mse, mse_prime);
    my_net.add(Dense::new(2, 4));
    my_net.add(Activation::new(relu, relu_prime));
    my_net.add(Dense::new(4, 1));
    let train = true;
    if train == true {
        let inputs = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]].into_iter().map(|i| i.to_vec().into()).collect();
        let outputs = vec![[0.0,], [1.0,], [1.0,], [0.0,]].into_iter().map(|i| i.to_vec().into()).collect();
        my_net.train(inputs, outputs, 0.03, 500);
        my_net.save("savefile.bin");
    } else {
        my_net.load("savefile.bin");
    }
    let test = true;
    if test == true {
        loop {
            let mut input = String::new();
            io::stdin().read_line(&mut input).expect("Failed to read line");
            println!("{:?}", my_net.forward_propagate(Vector::from_data( input.trim().split(' ').map(|a| a.parse::<f32>().expect("enter a number")).collect())));
        }
    }
}

