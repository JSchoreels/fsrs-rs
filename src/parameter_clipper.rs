use crate::{
    inference::Parameters,
    parameter_initialization::INIT_S_MAX,
    simulation::{D_MAX, D_MIN, S_MIN},
};
use burn::{
    module::Param,
    tensor::{Tensor, TensorData, backend::Backend},
};

const FSRS7_PARAM_LEN: usize = 35;

pub(crate) fn parameter_clipper<B: Backend>(
    parameters: Param<Tensor<B, 1>>,
    num_relearning_steps: usize,
    enable_short_term: bool,
) -> Param<Tensor<B, 1>> {
    let (id, val) = parameters.consume();
    let clipped = clip_parameters(
        &val.to_data().to_vec().unwrap(),
        num_relearning_steps,
        enable_short_term,
    );
    Param::initialized(
        id,
        Tensor::from_data(TensorData::new(clipped, val.shape()), &val.device()).require_grad(),
    )
}

fn clip_fsrs6_parameters(
    parameters: &mut [f32],
    num_relearning_steps: usize,
    enable_short_term: bool,
) {
    let w17_w18_ceiling = if parameters.len() > 14 && num_relearning_steps > 1 {
        (-(parameters[11].ln() + (2.0f32.powf(parameters[13]) - 1.0).ln() + parameters[14] * 0.3)
            / num_relearning_steps as f32)
            .max(0.01)
            .sqrt()
            .min(2.0)
    } else {
        2.0
    };
    let w19_floor = if enable_short_term { 0.01 } else { 0.0 };

    let clamps: [(f32, f32); 21] = [
        (S_MIN, INIT_S_MAX),
        (S_MIN, INIT_S_MAX),
        (S_MIN, INIT_S_MAX),
        (S_MIN, INIT_S_MAX),
        (D_MIN, D_MAX),
        (0.001, 4.0),
        (0.001, 4.0),
        (0.001, 0.75),
        (0.0, 4.5),
        (0.0, 0.8),
        (0.001, 3.5),
        (0.001, 5.0),
        (0.001, 0.25),
        (0.001, 0.9),
        (0.0, 4.0),
        (0.0, 1.0),
        (1.0, 6.0),
        (0.0, w17_w18_ceiling),
        (0.0, w17_w18_ceiling),
        (w19_floor, 0.8),
        (0.1, 0.8),
    ];

    parameters
        .iter_mut()
        .zip(clamps)
        .for_each(|(w, (low, high))| *w = w.clamp(low, high));
}

fn clip_fsrs7_parameters(parameters: &mut [f32]) {
    if parameters.len() < FSRS7_PARAM_LEN {
        return;
    }

    parameters[0] = parameters[0].clamp(S_MIN, INIT_S_MAX / 2.0);
    parameters[1] = parameters[1].clamp(parameters[0], INIT_S_MAX);
    parameters[2] = parameters[2].clamp(parameters[1], INIT_S_MAX);
    parameters[3] = parameters[3].clamp(parameters[2], INIT_S_MAX);

    parameters[4] = parameters[4].clamp(1.0, 10.0);
    parameters[5] = parameters[5].clamp(0.001, 4.0);
    parameters[6] = parameters[6].clamp(0.1, 4.0);

    parameters[7] = parameters[7].clamp(0.0, 4.0);
    parameters[8] = parameters[8].clamp(0.0, 1.2);
    parameters[9] = parameters[9].clamp(0.3, 3.0);
    parameters[10] = parameters[10].clamp(0.01, 1.5);
    parameters[11] = parameters[11].clamp(0.001, 0.9);
    parameters[12] = parameters[12].clamp(0.1, 1.0);
    parameters[13] = parameters[13].clamp(0.0, 3.5);
    parameters[14] = parameters[14].clamp(0.0, 1.0);
    parameters[15] = parameters[15].clamp(1.0, 7.0);

    parameters[16] = parameters[16].clamp(0.0, 4.0);
    parameters[17] = parameters[17].clamp(0.0, 2.0);
    parameters[18] = parameters[18].clamp(0.5, 6.0);
    parameters[19] = parameters[19].clamp(0.001, 1.5);
    parameters[20] = parameters[20].clamp(0.001, 2.0);
    parameters[21] = parameters[21].clamp(0.001, 1.0);
    parameters[22] = parameters[22].clamp(0.0, 5.0);
    parameters[23] = parameters[23].clamp(0.0, 1.0);
    parameters[24] = parameters[24].clamp(1.0, 7.0);

    parameters[25] = parameters[25].clamp(2.5, 15.0);
    parameters[26] = parameters[26].clamp(0.0, 1.0);

    parameters[27] = parameters[27].clamp(0.01, 0.25);
    parameters[28] = parameters[28].clamp(parameters[27], 0.95);
    parameters[29] = parameters[29].clamp(0.5, 0.85);
    parameters[30] = parameters[30].clamp(parameters[29], 0.99);
    parameters[31] = parameters[31].clamp(0.01, 1.0);
    parameters[32] = parameters[32].clamp(0.1, 1.0);
    parameters[33] = parameters[33].clamp(0.0, 0.9);
    parameters[34] = parameters[34].clamp(0.1, 1.1);
}

pub(crate) fn clip_parameters(
    parameters: &Parameters,
    num_relearning_steps: usize,
    enable_short_term: bool,
) -> Vec<f32> {
    let mut parameters = parameters.to_vec();
    if parameters.len() >= FSRS7_PARAM_LEN {
        clip_fsrs7_parameters(&mut parameters);
    } else {
        clip_fsrs6_parameters(&mut parameters, num_relearning_steps, enable_short_term);
    }
    parameters
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DEFAULT_PARAMETERS, test_helpers::Tensor};
    use burn::backend::ndarray::NdArrayDevice;
    static DEVICE: NdArrayDevice = NdArrayDevice::Cpu;

    #[test]
    fn test_parameter_clipper_works() {
        let tensor = Tensor::from_floats(
            [0.0, -1000.0, 1000.0, 0.0, 1000.0, -1000.0, 1.0, 0.25, -0.1],
            &DEVICE,
        );

        let param = parameter_clipper(Param::from_tensor(tensor), 1, true);
        let values = &param.to_data().to_vec::<f32>().unwrap();

        assert_eq!(
            values,
            &[0.001, 0.001, 100.0, 0.001, 10.0, 0.001, 1.0, 0.25, 0.0]
        );
    }

    #[test]
    fn test_parameter_clipper_works_with_num_relearning_steps() {
        use crate::test_helpers::TestHelper;
        let tensor = Tensor::from_floats(DEFAULT_PARAMETERS, &DEVICE);

        let param = parameter_clipper(Param::from_tensor(tensor), 2, true);
        let values = &param.to_data().to_vec::<f32>().unwrap();

        values[17..=19].assert_approx_eq([0.3072, 3.5875, 0.303]);
    }

    #[test]
    fn test_fsrs7_clipper_monotonic_bounds() {
        let mut params = vec![1000.0; 35];
        params[27] = -1.0;
        params[28] = 10.0;
        params[29] = 0.1;
        params[30] = 2.0;
        let clipped = clip_parameters(&params, 1, true);
        assert_eq!(clipped.len(), 35);
        assert!(clipped[1] >= clipped[0]);
        assert!(clipped[2] >= clipped[1]);
        assert!(clipped[3] >= clipped[2]);
        assert!(clipped[28] >= clipped[27]);
        assert!(clipped[30] >= clipped[29]);
    }
}
