import os
import sys
from argparse import Namespace

# Keep matplotlib cache writable in sandboxed environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SRS_BENCHMARK_ROOT = os.environ.get("SRS_BENCHMARK_ROOT", "../srs-benchmark")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, SRS_BENCHMARK_ROOT)

from config import Config  # noqa: E402
from models.fsrs_v7 import FSRS7  # noqa: E402


def build_config() -> Config:
    args = Namespace(
        processes=1,
        gpus=None,
        dev=False,
        max_user_id=None,
        partitions="none",
        recency=False,
        default=False,
        S0=False,
        two_buttons=False,
        data="../anki-revlogs-10k",
        secs=True,
        duration=False,
        no_test_same_day=False,
        no_train_same_day=False,
        equalize_test_with_non_secs=False,
        raw=False,
        file=False,
        plot=False,
        algo="FSRS-7",
        short=True,
        weights=False,
        train_equals_test=False,
        n_splits=2,
        batch_size=64,
        max_seq_len=32,
        torch_num_threads=1,
    )
    return Config(args)


def save_plot_1_forgetting_by_stability(model: FSRS7) -> None:
    t = torch.tensor(np.linspace(0.0, 14.0, 500), dtype=torch.float32)
    stabilities = [0.2, 0.5, 1.0, 3.0, 10.0, 30.0]

    plt.figure(figsize=(9, 5.5))
    for s in stabilities:
        s_tensor = torch.full_like(t, fill_value=s)
        r = model.forgetting_curve(
            t,
            s_tensor,
            -model.w[-8],
            -model.w[-7],
            model.w[-6],
            model.w[-5],
            model.w[-4],
            model.w[-3],
            model.w[-2],
            model.w[-1],
        ).detach().cpu().numpy()
        plt.plot(t.numpy(), r, label=f"S={s:g}")

    plt.title("FSRS-7 Forgetting Curve (Default Params) vs Stability")
    plt.xlabel("Elapsed time t (days)")
    plt.ylabel("Recall probability R(t)")
    plt.ylim(0.0, 1.02)
    plt.xlim(0.0, 14.0)
    plt.grid(True, alpha=0.25)
    plt.legend(title="Initial stability")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_forgetting_curve_by_stability.png", dpi=180)
    plt.close()


def _post_review_stability(model: FSRS7, base_s: float, base_d: float, review_delta_t: float, rating: int) -> float:
    X = torch.tensor([[review_delta_t, float(rating)]], dtype=torch.float32)
    state = torch.tensor([[base_s, base_d]], dtype=torch.float32)
    new_state = model.step(X, state)
    return float(new_state[0, 0].detach().cpu().item())


def save_plot_2_grade_conditioned(model: FSRS7) -> None:
    ratings = [1, 2, 3, 4]
    rating_names = {1: "Again", 2: "Hard", 3: "Good", 4: "Easy"}
    review_gaps = [0.2, 3.0]  # days
    base_s = 3.0
    base_d = 5.5
    t_future = torch.tensor(np.linspace(0.0, 21.0, 600), dtype=torch.float32)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)

    for ax, gap in zip(axes, review_gaps):
        for r in ratings:
            s_next = _post_review_stability(model, base_s, base_d, gap, r)
            s_tensor = torch.full_like(t_future, fill_value=s_next)
            curve = model.forgetting_curve(
                t_future,
                s_tensor,
                -model.w[-8],
                -model.w[-7],
                model.w[-6],
                model.w[-5],
                model.w[-4],
                model.w[-3],
                model.w[-2],
                model.w[-1],
            ).detach().cpu().numpy()
            ax.plot(
                t_future.numpy(),
                curve,
                label=f"{rating_names[r]} (new S={s_next:.2f})",
            )

        ax.set_title(f"After one review at Δt={gap}d")
        ax.set_xlabel("Future elapsed time t (days)")
        ax.set_xlim(0.0, 21.0)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Recall probability R(t)")
    axes[1].legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "FSRS-7 Grade-Conditioned Curves (from base state S=3.0, D=5.5)",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/02_grade_conditioned_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_plot_3_transition_coefficient(model: FSRS7) -> None:
    t = torch.tensor(np.linspace(0.0, 10.0, 500), dtype=torch.float32)
    coef = model.transition_function(t).detach().cpu().numpy()

    plt.figure(figsize=(9, 5.5))
    plt.plot(t.numpy(), coef, color="#004e98", linewidth=2)
    plt.title("FSRS-7 Long/Short Transition Coefficient")
    plt.xlabel("Elapsed time Δt (days)")
    plt.ylabel("coefficient(Δt)")
    plt.ylim(0.0, 1.02)
    plt.xlim(0.0, 10.0)
    plt.grid(True, alpha=0.25)

    w25 = float(model.w[25].detach().cpu().item())
    w26 = float(model.w[26].detach().cpu().item())
    plt.text(
        0.1,
        0.06,
        f"w25={w25:.3f}, w26={w26:.3f}\\ncoef(0)=1-w26={1-w26:.3f}",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_transition_coefficient.png", dpi=180)
    plt.close()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    config = build_config()
    model = FSRS7(config)

    save_plot_1_forgetting_by_stability(model)
    save_plot_2_grade_conditioned(model)
    save_plot_3_transition_coefficient(model)

    print("generated:")
    print(f"- {OUT_DIR}/01_forgetting_curve_by_stability.png")
    print(f"- {OUT_DIR}/02_grade_conditioned_curves.png")
    print(f"- {OUT_DIR}/03_transition_coefficient.png")


if __name__ == "__main__":
    main()
