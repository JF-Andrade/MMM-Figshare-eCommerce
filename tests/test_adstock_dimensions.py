import numpy as np
import pytensor
import pytensor.tensor as pt
from src.models.hierarchical_bayesian import geometric_adstock_pytensor

def test_adstock_short_sequence():
    """Test that adstock works when n_obs < l_max."""
    n_obs = 5
    n_channels = 2
    l_max = 10
    
    # Mock data
    x_val = np.random.randn(n_obs, n_channels).astype("float64")
    alpha_val = np.full((n_obs, n_channels), 0.5).astype("float64")
    territory_idx_val = np.zeros(n_obs, dtype="int32")
    
    # PyTensor variables
    x = pt.matrix("x")
    alpha = pt.matrix("alpha")
    territory_idx = pt.ivector("territory_idx")
    
    # Call the adstock function
    adstock_op = geometric_adstock_pytensor(
        x=x,
        alpha=alpha,
        territory_idx=territory_idx,
        l_max=l_max
    )
    
    # Compile the function
    f = pytensor.function([x, alpha, territory_idx], adstock_op)
    
    try:
        result = f(x_val, alpha_val, territory_idx_val)
        print(f"Success! Result shape: {result.shape}")
        assert result.shape == (n_obs, n_channels)
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")
        raise e

if __name__ == "__main__":
    test_adstock_short_sequence()
