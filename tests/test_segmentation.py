from src.preprocessing.segmentation import SegmentConfig, split_nominal_excitation
import pandas as pd


def test_split_nominal_excitation():
    df = pd.DataFrame(
        {
            "t": [0, 1, 2, 3],
            "theta": [0.0, 0.1, 0.2, 0.3],
            "omega": [0.0, 0.2, 0.1, 0.0],
            "u": [0.0, 0.01, 0.2, -0.3],
        }
    )
    nominal, exc = split_nominal_excitation(df, SegmentConfig(zero_input_threshold=0.05))
    assert len(nominal) == 2
    assert len(exc) == 2
