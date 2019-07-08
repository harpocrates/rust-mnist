extern crate image;
extern crate rand;
extern crate rand_distr;
extern crate ndarray;
extern crate simple_server;

mod digit_image;
mod network;
mod cost;

use std::io;

// TODO:
//
// Plot and log the:
//   - total cost function for training/testing epochs
//   - accuracy rate for training/testing epochs
//
// Support interactive stuff with browser:
//   - parameters
//   - loading/saving models
//   - training

fn main() -> Result<(), io::Error> {
    let training = digit_image::read_data("train")?;
    let testing = digit_image::read_data("t10k")?;
    println!("Read in the training and testing data sets.");

    for i in 0..10 {
        let im = &training[i];
        im.save_to_file(format!("image{}_{}.png", i, im.label))?;
    }
    println!("Wrote out a the first ten images in the training set.");

    let training: Vec<_> = training.iter().map(|im| im.into()).collect();
    let testing: Vec<_>  = testing .iter().map(|im| im.into()).collect();

    let mut network = network::Network::new(vec![784, 100, 10]);

    println!("Training is under progress...");
  //  network.stochastic_gradient_descent::<cost::QuadraticCost>(30, 10, training, 0.15, 0.0, Some(&testing));
    network.stochastic_gradient_descent::<cost::CrossEntropyCost>(30, 10, training, 0.05, 5.0, Some(&testing));

    println!("Web server is up!");
    let server = simple_server::Server::new(move |request, mut response| {
        let uri = request.uri();
        Ok(match (request.method(), uri.path(), uri.query()) {
            (&simple_server::Method::GET, "/", None) |
            (&simple_server::Method::GET, "/index.html", None) =>
                response.body(include_bytes!("index.html").to_vec())?,

            (&simple_server::Method::GET, "/whats_my_number", Some(hex_str)) => {
                let image = &digit_image::Image::from_hex_string(hex_str, 0);
                image.save_to_file(format!("image_last.png"))?;
                let data: network::LabelledInput = image.into();
                let output = network.apply_all(data.input);

                response
                    .header("Content-Type", "text/plain")
                    .body(format!("Ranked outputs: {:?}", output).as_bytes().to_vec())?
            }

            _ => response
                    .status(500)
                    .body("Unknown".as_bytes().to_vec())?
        })
    });

    server.listen("127.0.0.1", "8080");
}

