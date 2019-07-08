#![allow(dead_code)]

use ndarray::{Array1};
use super::network::sigmoid_derivative;

/// TODO: explain properties of cost, and motivation
pub trait Cost {
    /// Determine the cost based on the activation and expected output
    fn cost(
        activation: &Array1<f64>,
        expected: &Array1<f64>,
    ) -> f64;

    /// Error delta from the output layer
    fn  delta(
        z: &Array1<f64>,           // `activation = σ(z)` 
        activation: &Array1<f64>,
        expected: &Array1<f64>,
    ) -> Array1<f64>;
}

pub enum QuadraticCost { }

impl Cost for QuadraticCost {
    fn cost(
        a: &Array1<f64>,
        y: &Array1<f64>
    ) -> f64 {
        let diff = &(y - a);
        0.5 * diff.dot(diff)
    }

    fn delta(
        z: &Array1<f64>,
        a: &Array1<f64>,
        y: &Array1<f64>,
    ) -> Array1<f64> {
        (a - y) * sigmoid_derivative(z)
    }
}

pub enum CrossEntropyCost { }

impl Cost for CrossEntropyCost {
    fn cost(
        a: &Array1<f64>,
        y: &Array1<f64>
    ) -> f64 {
        let mut result = 0.0;
        for (a,y) in a.iter().zip(y.iter()) {
            let potential = -y * a.ln() - (1.0  - y) * (1.0 -  a).ln();
            result += if potential.is_nan() { 0.0 } else { potential };
        }
        result
    }

    fn delta(
        _z: &Array1<f64>,
        a: &Array1<f64>,
        y: &Array1<f64>,
    ) -> Array1<f64> {
        a - y
    }

}
