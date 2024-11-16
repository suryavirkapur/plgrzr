use anyhow::Result;
use std::{collections::HashMap, path::{Path, PathBuf}};
use tch::{Device, Tensor};

pub struct HandwritingDataset {
    samples: Vec<(PathBuf, usize)>,
    transforms: transforms::ImageTransforms,
}

impl HandwritingDataset {
    pub fn new<P: AsRef<Path>>(data_dir: P, device: Device) -> Result<Self> {
        let mut samples = Vec::new();
        let mut author_map = HashMap::new();

        for entry in std::fs::read_dir(data_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let author = entry.file_name();
                let author_id = author_map.len();
                author_map.insert(author.clone(), author_id);

                for img in std::fs::read_dir(entry.path())? {
                    let img = img?;
                    if img.file_type()?.is_file() {
                        samples.push((img.path(), author_id));
                    }
                }
            }
        }

        Ok(Self {
            samples,
            transforms: transforms::ImageTransforms::new(device),
        })
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn get_batch(&self, indices: &[usize]) -> Result<(Tensor, Tensor)> {
        let mut images = Vec::with_capacity(indices.len());
        let mut labels = Vec::with_capacity(indices.len());

        for &idx in indices {
            let (path, author_id) = &self.samples[idx];
            let img = image::open(path)?;
            let tensor = self.transforms.process(img)?;
            images.push(tensor);
            labels.push(*author_id as i64);
        }

        let images = Tensor::stack(&images, 0);
        let labels = Tensor::of_slice(&labels);

        Ok((images, labels))
    }
}
