import os

import numpy as np
from absl import app, flags

from examples.states.train_diffusion_offline import call_main
from launcher.hyperparameters import set_hyperparameters


FLAGS = flags.FLAGS
flags.DEFINE_integer("variant", 0, "Logging interval.")


def main(_):
    constant_parameters = dict(
        project="offline_schedule_final",
        experiment_name="ddpm_iql",
        max_steps=3000001,  # Actor takes two steps per critic step
        batch_size=512,
        eval_episodes=50,
        log_interval=1000,
        eval_interval=250000,
        save_video=False,
        filter_threshold=None,
        take_top=None,
        online_max_steps=0,
        unsquash_actions=False,
        normalize_returns=True,
        training_time_inference_params=dict(
            N=64,
            clip_sampler=True,
            M=0,
        ),
        rl_config=dict(
            model_cls="DDPMIQLLearner",
            actor_lr=3e-4,
            critic_lr=3e-4,
            value_lr=3e-4,
            T=5,
            N=64,
            M=0,
            actor_dropout_rate=0.1,
            actor_num_blocks=3,
            decay_steps=int(3e6),
            actor_layer_norm=True,
            value_layer_norm=True,
            actor_tau=0.001,
            beta_schedule="vp",
        ),
    )

    sweep_parameters = dict(
        seed=list(range(10)),
        env_name=[
            "Walker2d-v4",
            "Walker2d-replay-v4",
            "Walker2d-expert-v4",
            "halfcheetah-medium-v4",
            "halfcheetah-medium-replay-v4",
            "halfcheetah-medium-expert-v4",
            "hopper-medium-v4",
            "hopper-medium-replay-v4",
            "hopper-medium-expert-v4",
            "antmaze-umaze-v4",
            "antmaze-umaze-diverse-v4",
            "antmaze-medium-diverse-v4",
            "antmaze-medium-play-v4",
            "antmaze-large-diverse-v4",
            "antmaze-large-play-v4",
        ],
    )

    variants = [constant_parameters]
    name_keys = ["experiment_name", "env_name"]
    variants = set_hyperparameters(sweep_parameters, variants, name_keys)

    inference_sweep_parameters = dict(
        N=[16, 64, 256],
        clip_sampler=[True],
        M=[0],
    )

    inference_variants = [{}]
    name_keys = []
    inference_variants = set_hyperparameters(
        inference_sweep_parameters, inference_variants
    )

    filtered_variants = []
    for variant in variants:
        #variant["rl_config"]["T"] = variant["T"]
        #variant["rl_config"]["beta_schedule"] = variant["beta_schedule"]
        variant["inference_variants"] = inference_variants

        if "antmaze" in variant["env_name"]:
            variant["rl_config"]["critic_hyperparam"] = 0.9
        else:
            variant["rl_config"]["critic_hyperparam"] = 0.7

        filtered_variants.append(variant)

    print(len(filtered_variants))
    variant = filtered_variants[FLAGS.variant]
    print(FLAGS.variant)
    call_main(variant)


if __name__ == "__main__":
    app.run(main)
