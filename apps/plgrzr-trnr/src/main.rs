use anyhow::Result;
mod config;
mod dataset;
mod model;
mod trainer;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let config = config::TrainingConfig::default();

    let mut trainer = trainer::HandwritingTrainer::new(config)?;

    trainer.train()?;

    Ok(())
}
