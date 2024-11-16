use std::{fs, path::PathBuf};
use tch::Device;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct TrainingConfig {
    pub data_dir: PathBuf,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub epochs: usize,
    pub device: Device,
    pub model_save_path: PathBuf,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        fs::create_dir_all("../../data/models").unwrap();
        Self {
            data_dir: PathBuf::from("data/handwriting_dataset"),
            batch_size: 32,
            learning_rate: 0.001,
            epochs: 100,
            device: Device::Cuda,
            model_save_path: PathBuf::from("../../data/models/handwriting_model.pt"),
        }
    }
}
