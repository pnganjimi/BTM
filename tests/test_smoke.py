import torch
from torch.utils.data import DataLoader, TensorDataset

from btm.models import get_model
from btm.utils import learn_low_loss_bezier_curve, state_from_curve


def test_model_output_shapes():
    assert get_model("eicu", device="cpu")(torch.randn(2, 402)).shape == (2, 1)
    assert get_model("mimic3_ihm", device="cpu")(torch.randn(2, 48, 60)).shape == (2, 1)
    assert get_model("mimic3_ph", device="cpu")(torch.randn(2, 48, 60)).shape == (2, 25)


def test_small_bezier_path_cpu():
    x = torch.randn(8, 402)
    y = (torch.rand(8) > 0.5).float()
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    model_a = get_model("eicu", device="cpu")
    model_b = get_model("eicu", device="cpu")
    model = get_model("eicu", device="cpu")

    theta_a = model_a.state_dict()
    theta_b = model_b.state_dict()

    result = learn_low_loss_bezier_curve(
        model,
        loader,
        theta_a,
        theta_b,
        torch.nn.BCEWithLogitsLoss(),
        device="cpu",
        num_mc_samples=1,
        num_steps=1,
        lr=1e-3,
        verbose=False,
    )

    state = state_from_curve(
        model, theta_a, theta_b, result["theta_C"], 0.5, "cpu"
    )
    assert set(state) == set(model.state_dict())
