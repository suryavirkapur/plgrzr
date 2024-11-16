use tch::{nn, Tensor};

#[derive(Debug)]
pub struct SiameseNetwork {
    feature_extractor: nn::Sequential,
    fc: nn::Sequential,
}

impl SiameseNetwork {
    pub fn new(vs: &nn::Path) -> Self {
        let feature_extractor = nn::seq()
            .add(nn::conv2d(vs, 1, 64, 10, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::max_pool2d(2, 2, 0, 1, true))
            .add(nn::conv2d(vs, 64, 128, 7, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::max_pool2d(2, 2, 0, 1, true))
            .add(nn::conv2d(vs, 128, 128, 4, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::max_pool2d(2, 2, 0, 1, true))
            .add(nn::conv2d(vs, 128, 256, 4, Default::default()))
            .add_fn(|x| x.relu());

        let fc = nn::seq()
            .add(nn::linear(vs, 256 * 6 * 6, 4096, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs, 4096, 1024, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs, 1024, 256, Default::default()));

        Self {
            feature_extractor,
            fc,
        }
    }

    pub fn forward_one(&self, x: &Tensor) -> Tensor {
        let x = self.feature_extractor.forward(x);
        let x = x.view([-1, 256 * 6 * 6]);
        self.fc.forward(&x)
    }

    pub fn forward(&self, input1: &Tensor, input2: &Tensor) -> (Tensor, Tensor) {
        let output1 = self.forward_one(input1);
        let output2 = self.forward_one(input2);
        (output1, output2)
    }
}
