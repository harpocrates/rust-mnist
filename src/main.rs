extern crate image;
extern crate rand;
extern crate rand_distr;
extern crate ndarray;
extern crate simple_server;

use std::io;
use std::io::prelude::*;
use std::fs::File;
use std::path::Path;
use std::ops::{MulAssign, AddAssign};

fn main() -> Result<(), io::Error> {
    let training = read_data("train")?;
    let testing = read_data("t10k")?;
    println!("Read in the training and testing data sets.");

    for i in 0..10 {
        let im = &training[i];
        im.save_to_file(format!("image{}_{}.png", i, im.label))?;
    }
    println!("Wrote out a the first ten images in the training set.");
    
    let training: Vec<_> = training.iter().map(|im| im.to_labelled()).collect();
    let testing: Vec<_>  = testing .iter().map(|im| im.to_labelled()).collect();
    
    let mut network = Network::new(vec![784, 30, 10]);

    println!("Training is under progress...");
    network.stochastic_gradient_descent(30, 10, training, 3.0, Some(&testing));

    Ok(())
}

fn sigmoid(z: &ndarray::Array1<f64>) -> ndarray::Array1<f64> {
    1.0 / (1.0 + z.map(|x| (-x).exp()))
}

fn sigmoid_derivative(z: &ndarray::Array1<f64>) -> ndarray::Array1<f64> {
    let sig = sigmoid(z);
    sig.clone() * (1.0 - sig)
}

#[derive(Debug)]
struct Network {
    layers: Vec<Layer>, 
}

/// A layer of size `n`, working of of a layer  of size `m`
#[derive(Debug)]
struct Layer {
    weights: ndarray::Array2<f64>,  // `m` by `n`
    biases: ndarray::Array1<f64>,   // `n`
}

impl Layer {
    pub fn input_size(&self) -> usize {
        self.weights.shape()[1]
    }

    pub fn output_size(&self) -> usize {
        self.weights.shape()[0]
    }
}

type Input = ndarray::Array1<f64>;
type LabelledInput = (ndarray::Array1<f64>, usize);

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
    pub fn input_size(&self) -> usize {
        self.layers[0].input_size()
    }
    
    pub fn output_size(&self) -> usize {
        self.layers[self.layers.len() - 1].output_size()
    }


    /// Create a new network with layers of the given sizes
    fn new(sizes: Vec<usize>) -> Network {
        use rand::prelude::*;
        
        let mut layers = Vec::new();
        let mut rng = thread_rng();

        for (&x, &y)  in sizes.iter().zip(sizes.iter().skip(1)) {
            let weights = ndarray::Array::from_shape_fn(
                (y, x),
                |_| rng.sample(rand_distr::StandardNormal),
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
    fn feed_forward(&self, input: Input) -> ndarray::Array1<f64> {
        let mut a: ndarray::Array1<f64> = input;
        for Layer { weights, biases } in &self.layers {
            a = sigmoid(&(weights.dot(&a) + biases))
        }
        a
    }

    /// On how many elements from the training set does this return the right answer?
    ///
    /// The neural net's answer is whichever entry is largest in the final layer.
    fn evaluate(&self, labelled: &[LabelledInput]) -> usize {
        let mut count = 0;
        
        for (input, expected_output) in labelled {
             
            let computed_output = self
                .feed_forward(input.clone())
                .indexed_iter()
                .into_iter()
                .fold(
                    (std::f64::NEG_INFINITY, std::usize::MAX),
                    |(f1,ix1), (ix2,&f2)| if f2 > f1 { (f2,ix2) } else { (f1,ix1) }
                );
            if computed_output.1 == *expected_output {
                count += 1;
            }
        }

        count
    }

    fn stochastic_gradient_descent(
        &mut self,
        epochs: usize,
        mini_batch_size: usize,
        mut training: Vec<LabelledInput>,
        eta: f64,
        test: Option<&[LabelledInput]>, // used for debug output
    ) {
        use rand::prelude::*;
        let mut rng = thread_rng();

        // How many times over will we look at every bit of data?
        for i in 0..epochs {

            // Shuffle the training data into different chunks and process these
            training.shuffle(&mut rng);
            for batch in training.chunks(mini_batch_size) {
                self.stochastic_gradient_descent_batch(batch, eta);
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
    fn stochastic_gradient_descent_batch(&mut self, training_batch: &[LabelledInput], eta: f64) {
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

        let factor = - (eta / (training_batch.len() as f64));
        for labelled_input in training_batch {
            nabla += Network { layers: self.backpropagate(labelled_input) };
        }

        nabla *= factor; 
        *self += nabla;
    }

    /// Returns the gradient of the cost function W.R.T. all weights and biases
    fn backpropagate(&self, labelled_input: &LabelledInput) -> Vec<Layer> {
        let (input, expected_output) =  labelled_input;
        let y = ndarray::Array::from_shape_fn(
            self.output_size(),
            |i| if i == *expected_output { 1.0 } else { 0.0 },
        );

        /* input + feed forward:
         *
         * a^1 = input
         * z^l = w^l a^{l-1} + b^l                  for l = 2, 3, ..., L
         * a^l = σ(z^l)                             for l = 2, 3, ..., L
         */
        let mut activation = input.clone();
        let mut activations = vec![];
        let mut zs = vec![];
        for layer in &self.layers {
            let z = layer.weights.dot(&activation) + &layer.biases;
            zs.push(z.clone());
            let prev_activation = std::mem::replace(&mut activation, sigmoid(&z));
            activations.push(prev_activation)
        }
        activations.push(activation.clone());

        /* output error + backpropagate error:
         *
         * δ^L = ∇aC ⊙ σ'(z^L)
         *     = (a^L - y) ⊙ σ'(z^L)                (since C = 1/2 (y -  a^L)^2)
         * 
         * δ^l = ((w^{l+1})^T δ^{l+1}) ⊙ σ′(z^l)    for l = L-1, L-2, ..., 2
         */ 
        let mut error: ndarray::Array1<f64> = (activation - y) * (&sigmoid_derivative(&zs[zs.len() - 1]));
        let mut errors = vec![error.clone()];

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
        let nabla: Vec<Layer> = activations.iter().zip(errors.iter())
            .map(|(activation, error): (&ndarray::Array1<f64>, &ndarray::Array1<f64>)| {
                Layer {
                    biases: error.clone(),
                    weights: error.clone().insert_axis(ndarray::Axis(1)).dot(&activation.clone().insert_axis(ndarray::Axis(0))),
                }
            })
            .collect();

        nabla
    }
}

struct Image {
    label: u8,      // actual number
    data: [u8; 784] // Images are 28 * 28, row-wise
}

impl Image {

    /// For debug purposes, save the image to a file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let mut img_buffer = image::ImageBuffer::new(28, 28);
        for (x, y, pixel) in img_buffer.enumerate_pixels_mut() {
            let darkness = self.data[(y * 28 + x) as usize];
            *pixel = image::Rgb([255 - darkness, 255 - darkness, 255 - darkness]);
        }
        img_buffer.save(path)
    }

    pub fn to_labelled(&self) -> LabelledInput {
        let input = ndarray::arr1(&self.data).map(|u| *u as f64 / 255.0);
        let expected_output = self.label as usize;
        (input, expected_output)
    }
}


fn read_data(data_set: &str) -> Result<Vec<Image>, io::Error> {
    let images = format!("mnist/{}-images-idx3-ubyte", data_set);
    let labels = format!("mnist/{}-labels-idx1-ubyte", data_set);

    let mut images_file = File::open(images)?;
    let mut labels_file = File::open(labels)?;

    fn read_int<R: Read>(reader: &mut R) -> Result<u32, io::Error> {
        let mut buf: [u8; 4] = [0,0,0,0];
        reader.read_exact(&mut  buf)?;
        Ok(u32::from_be_bytes(buf))
    }

    let label_magic_constant = read_int(&mut labels_file)?;
    let label_num_items = read_int(&mut labels_file)?;
    
    let image_magic_constant = read_int(&mut images_file)?;
    let image_num_items = read_int(&mut images_file)?;
    let rows = read_int(&mut images_file)?;
    let cols = read_int(&mut images_file)?;
    
    assert_eq!(label_magic_constant, 2049);
    assert_eq!(image_magic_constant, 2051);
    assert_eq!(label_num_items, image_num_items);
    assert_eq!(rows, 28);
    assert_eq!(cols, 28);

    (0..image_num_items)
        .map(|_| {
            let mut label_data = [0u8]; 
            let mut data = [0u8; 784];

            labels_file.read_exact(&mut label_data)?;
            images_file.read_exact(&mut data)?;

            Ok(Image { label: label_data[0], data })
        })
        .collect()
}
