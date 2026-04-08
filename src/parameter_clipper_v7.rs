use crate::{parameter_initialization::INIT_S_MAX, simulation::S_MIN};

const FSRS7_PARAM_LEN: usize = 35;

pub(crate) fn clip_fsrs7_parameters(parameters: &mut [f32]) {
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
