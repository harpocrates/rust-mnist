use super::network;

use std::io;
use std::path::Path;
use std::fs::File;
use std::io::Read;

/// A 28 by 28 pixel image of a digit, annotated with a label describing the
/// digit. Data is structured row-wwise, with one byte per pixel for indicating
/// how dark the pixel is.
pub struct Image {
    pub label: u8,      // actual number
    pub data: [u8; 784] // Images are 28 * 28, row-wise
}

impl Image {

    /// Extract *some* image from the hex string. The idea is the data should be
    /// encoded with each byte being 2 hex characters.
    pub fn from_hex_string(hex_str: &str, label: u8) -> Image {
        let mut image = Image { label, data: [0; 784] };

        for (ix, chunk) in hex_str.chars().collect::<Vec<_>>().chunks(2).enumerate() {
            let c0 = chunk[0].to_digit(16).map_or(0, |d| d as u8);
            let c1 = chunk[1].to_digit(16).map_or(0, |d| d as u8);
            image.data[ix] = 16 * c0 + c1;
        }

        image
    }

    /// For debug purposes, save the image to a file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let mut img_buffer = image::ImageBuffer::new(28, 28);
        for (x, y, pixel) in img_buffer.enumerate_pixels_mut() {
            let darkness = self.data[(y * 28 + x) as usize];
            *pixel = image::Rgb([255 - darkness, 255 - darkness, 255 - darkness]);
        }
        img_buffer.save(path)
    }
}

impl From<&Image> for network::LabelledInput {
    fn from(image: &Image) -> network::LabelledInput {
        let slice: &[u8] = &image.data;
        let input = ndarray::ArrayView::from(slice).mapv(|u| u as f64 / 255.0);
        let output = image.label as usize;
        network::LabelledInput { input, output }
    }
}

/// Read images and labels data for the [MNIST database of handwritten digits][0].
/// That link includes a detailed description of how images and labels are encoded
/// in the data files.
///
/// [0]: http://yann.lecun.com/exdb/mnist/
pub fn read_data(data_set: &str) -> Result<Vec<Image>, io::Error> {
    let images = format!("mnist/{}-images-idx3-ubyte", data_set);
    let labels = format!("mnist/{}-labels-idx1-ubyte", data_set);

    let mut images_file = File::open(images)?;
    let mut labels_file = File::open(labels)?;

    // Pull one unsigned 32-bit number off the front of the reader
    fn read_int(reader: &mut impl Read) -> Result<u32, io::Error> {
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
