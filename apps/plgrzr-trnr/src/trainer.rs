use anyhow::Result;
use tch::{
    nn::{self, OptimizerConfig},
    Device, Tensor,
};
use tracing::{info, warn};

use crate::{config::TrainingConfig, dataset::HandwritingDataset, model::SiameseNetwork};

pub struct HandwritingTrainer {
    config: TrainingConfig,
    model: SiameseNetwork,
    dataset: HandwritingDataset,
    optimizer: nn::Optimizer,
}

impl HandwritingTrainer {
    pub fn new(config: TrainingConfig) -> Result<Self> {
        let vs = nn::VarStore::new(config.device);
        let model = SiameseNetwork::new(&vs.root());
        let dataset = HandwritingDataset::new(&config.data_dir, config.device)?;
        let optimizer = nn::Adam::default().build(&vs, config.learning_rate)?;

        Ok(Self {
            config,
            model,
            dataset,
            optimizer,
        })
    }

    fn contrastive_loss(&self, output1: &Tensor, output2: &Tensor, label: &Tensor) -> Tensor {
        let euclidean_distance = output1.sub(output2).pow(2.0).sum1(&[-1], false).sqrt();
        let loss = label * euclidean_distance.pow(2.0)
            + (1.0 - label)
                * (2.0 - euclidean_distance)
                    .clamp(0.0, f64::INFINITY)
                    .pow(2.0);
        loss.mean(Kind::Float)
    }

    pub fn train(&mut self) -> Result<()> {
        info!("Starting training...");

        for epoch in 0..self.config.epochs {
            let mut total_loss = 0.0;
            let num_batches = self.dataset.len() / self.config.batch_size;

            for batch_idx in 0..num_batches {
                // Generate batch indices
                let indices: Vec<_> = (0..self.config.batch_size)
                    .map(|_| rand::random::<usize>() % self.dataset.len())
                    .collect();

                // Get batch data
                let (images, labels) = self.dataset.get_batch(&indices)?;

                // Split into pairs
                let (anchor, paired) = images.split_at(self.config.batch_size / 2, 0);
                let (labels1, labels2) = labels.split_at(self.config.batch_size / 2, 0);
                let target = labels1.eq(&labels2).to_kind(Kind::Float);

                // Forward pass
                let (output1, output2) = self.model.forward(&anchor, &paired);
                let loss = self.contrastive_loss(&output1, &output2, &target);

                // Backward pass
                self.optimizer.zero_grad();
                loss.backward();
                self.optimizer.step();

                total_loss += f64::from(&loss);

                if batch_idx % 10 == 0 {
                    info!(
                        "Epoch: {}/{}, Batch: {}/{}, Loss: {:.6}",
                        epoch + 1,
                        self.config.epochs,
                        batch_idx + 1,
                        num_batches,
                        loss.double_value(&[])
                    );
                }
            }

            let avg_loss = total_loss / num_batches as f64;
            info!("Epoch: {}, Average Loss: {:.6}", epoch + 1, avg_loss);

            // Save model checkpoint
            if (epoch + 1) % 10 == 0 {
                self.save_checkpoint(epoch + 1, avg_loss)?;
            }
        }

        Ok(())
    }

    fn save_checkpoint(&self, epoch: usize, loss: f64) -> Result<()> {
        let checkpoint = Checkpoint {
            epoch,
            model_state: self.model.state_dict(),
            optimizer_state: self.optimizer.state_dict(),
            loss,
        };

        let path = self
            .config
            .model_save_path
            .with_file_name(format!("checkpoint_epoch_{}.pt", epoch));

        checkpoint.save(&path)?;
        info!("Saved checkpoint to {:?}", path);

        Ok(())
    }
}
