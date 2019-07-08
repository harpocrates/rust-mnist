use std::ops::MulAssign;
use std::ops::AddAssign;
use std::cmp::Ordering;

use ndarray::Array1;
use super::cost::Cost;

pub fn sigmoid(z: &Array1<f64>) -> Array1<f64> {
    1.0 / (1.0 + z.map(|x| (-x).exp()))
}

pub fn sigmoid_derivative(z: &Array1<f64>) -> Array1<f64> {
    let sig = sigmoid(z);
    sig.clone() * (1.0 - sig)
}

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
}

/// A layer of size `n`, working of of a layer  of size `m`
#[derive(Debug)]
struct Layer {
    weights: ndarray::Array2<f64>,  // `m` by `n`
    biases: Array1<f64>,   // `n`
}

impl Layer {
    pub fn input_size(&self) -> usize {
        self.weights.shape()[1]
    }

    pub fn output_size(&self) -> usize {
        self.weights.shape()[0]
    }
}

type Input = Array1<f64>;
pub struct LabelledInput {
    pub input: Array1<f64>,
    pub output: usize,
}

impl AddAssign for Layer {
    fn  add_assign(&mut self, other: Layer) {
        self.weights += &other.weights;
        self.biases += &other.biases;
    }
}

impl AddAssign for Network {
    fn add_assign(&mut self, other: Network) {
        for (l1, l2) in self.layers.iter_mut().zip(other.layers.into_iter()) {
            *l1 += l2;
        }
    }
}

impl MulAssign<f64> for Layer {
    fn mul_assign(&mut self, other: f64) {
        self.weights *= other;
        self.biases *= other;
    }
}

impl MulAssign<f64> for Network {
    fn mul_assign(&mut self, other: f64) {
        for l in self.layers.iter_mut() {
            *l *= other;
        }
    }
}


impl Network {
    #[allow(dead_code)]
    pub fn input_size(&self) -> usize {
        self.layers[0].input_size()
    }

    #[allow(dead_code)]
    pub fn output_size(&self) -> usize {
        self.layers[self.layers.len() - 1].output_size()
    }


    /// Create a new network with layers of the given sizes
    pub fn new(sizes: Vec<usize>) -> Network {
        use rand::prelude::*;

        let mut layers = Vec::new();
        let mut rng = thread_rng();

        for (&x, &y)  in sizes.iter().zip(sizes.iter().skip(1)) {
            let weight_distr = rand_distr::Normal::new(0.0, (x as f64).sqrt().recip()).unwrap();
            let weights = ndarray::Array::from_shape_fn(
                (y, x),
                |_| rng.sample(weight_distr),
            );
            
            let biases = ndarray::Array::from_shape_fn(
                y,
                |_| rng.sample(rand_distr::StandardNormal),
            );

            layers.push(Layer { weights, biases })
        }

        Network { layers }
    }

    /// Applies each of the layers in the network to the input
    fn feed_forward(&self, input: Input) -> Array1<f64> {
        let mut a: Array1<f64> = input;
        for Layer { weights, biases } in &self.layers {
            a = sigmoid(&(weights.dot(&a) + biases))
        }
        a
    }

    /// Like feedforward, but returns the index of the largets entry
    fn apply(&self, input: Input) -> usize {
        self.feed_forward(input)
            .indexed_iter()
            .into_iter()
            .fold(
                (std::f64::NEG_INFINITY, std::usize::MAX),
                |(f1,ix1), (ix2,&f2)| if f2 > f1 { (f2,ix2) } else { (f1,ix1) }
            )
            .1
    }

    /// Like feedforward, but returns the indices of entries of decreasing size
    pub fn apply_all(&self, input: Input) -> Vec<usize> {
        let mut ixed = self
            .feed_forward(input)
            .indexed_iter()
            .map(|(ix,f)| (ix,*f))
            .collect::<Vec<_>>();

        ixed.sort_by(|(_,f1), (_,f2)|
            if f1 < f2 {
                Ordering::Less
            } else if f1 == f2 {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        );

        ixed.into_iter().map(|(ix, _)| ix).rev().collect()
    }

    /// On how many elements from the training set does this return the right answer?
    ///
    /// The neural net's answer is whichever entry is largest in the final layer.
    fn evaluate(&self, labelled: &[LabelledInput]) -> usize {
        let mut count = 0;

        for LabelledInput { input, output } in labelled {
            if self.apply(input.clone()) == *output {
                count += 1;
            }
        }

        count
    }

    pub fn stochastic_gradient_descent<C: Cost>(
        &mut self,
        epochs: usize,
        mini_batch_size: usize,
        mut training: Vec<LabelledInput>,
        eta: f64,                       // learning rate
        lambda: f64,                    // factor for regularization
        test: Option<&[LabelledInput]>, // used for debug output
    ) {
        use rand::prelude::*;
        let mut rng = thread_rng();

        // How many times over will we look at every bit of data?
        for i in 0..epochs {

            // Shuffle the training data into different chunks and process these
            training.shuffle(&mut rng);
            for batch in training.chunks(mini_batch_size) {
                self.stochastic_gradient_descent_batch::<C>(batch, training.len(), eta, lambda);
            }

            // Display training progress
            if let Some(test) = test {
                println!("Epoch {} ({} / {})", i, self.evaluate(test), test.len());
            } else {
                println!("Epoch {}", i);
            }
        }

    }

    /// One iteration of stochastic gradient descent on a small batch of labelled input.
    /// `eta` is the learning rate.
    fn stochastic_gradient_descent_batch<C: Cost>(
        &mut self,
        training_batch: &[LabelledInput],
        training_data_size: usize,
        eta: f64,
        lambda: f64,
    ) {
        // How much should `nabla` affect the network?
        let factor = - (eta / (training_batch.len() as f64));
        
        // How much should the weights decay?
        let decay = 1.0 - eta * (lambda / training_data_size as f64);
        
        // Build up `nabla`, the gradient
        let mut nabla = Network {
            layers: self.layers
                .iter()
                .map(|l| {
                    Layer {
                        weights: ndarray::Array::zeros((l.output_size(), l.input_size())),
                        biases: ndarray::Array::zeros(l.output_size()),
                    }
                })
                .collect()
        };
        for labelled_input in training_batch {
            nabla += Network {
                 layers: self.backpropagate::<C>(labelled_input),
            };
        }


        // Decay the weights...
        for layer in &mut self.layers {
            layer.weights *= decay;
        }

        // Inch up the hill... 
        nabla *= factor;
        *self += nabla;
    }

    /// Returns the gradient of the cost function W.R.T. all weights and biases
    fn backpropagate<C: Cost>(&self, labelled: &LabelledInput) -> Vec<Layer> {
        let y = ndarray::Array::from_shape_fn(
            self.output_size(),
            |i| if i == labelled.output { 1.0 } else { 0.0 },
        );

        /* input + feed forward:
         *
         * a^1 = input
         * z^l = w^l a^{l-1} + b^l                  for l = 2, 3, ..., L
         * a^l = σ(z^l)                             for l = 2, 3, ..., L
         */
        let mut activation = labelled.input.clone();
        let mut activations = vec![];
        let mut zs = vec![];
        for layer in &self.layers {
            let z = layer.weights.dot(&activation) + &layer.biases;
            zs.push(z.clone());
            let prev_activation = std::mem::replace(&mut activation, sigmoid(&z));
            activations.push(prev_activation)
        }

        /* output error + backpropagate error:
         *
         * δ^L = ∇aC ⊙ σ'(z^L)
         *     = (a^L - y) ⊙ σ'(z^L)                (since C = 1/2 (y -  a^L)^2)
         *
         * δ^l = ((w^{l+1})^T δ^{l+1}) ⊙ σ′(z^l)    for l = L-1, L-2, ..., 2
         */
        let mut error: Array1<f64> = C::delta(&zs[zs.len() - 1], &activation, &y);
        let mut errors = vec![error.clone()];
        activations.push(activation);

        for (layer, z) in self.layers.iter().skip(1).zip(zs.iter()).rev() {
            error = (layer.weights.t().dot(&error)) * (&sigmoid_derivative(z));
            errors.push(error.clone());
        }
        errors.reverse();

        /* output
         *
         * ∂C/∂w^l_{jk} = a^{l−1}_k δ^l_j
         * ∂C/∂b^l_j    = δ^l_j
         */
        let nabla: Vec<Layer> = activations.into_iter().zip(errors.into_iter())
            .map(|(activation, error): (Array1<f64>, Array1<f64>)| {
                let weights = error.clone().insert_axis(ndarray::Axis(1)).dot(&activation.insert_axis(ndarray::Axis(0)));
                Layer { biases: error, weights }
            })
            .collect();

        nabla
    }
}



