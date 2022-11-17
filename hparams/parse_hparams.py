from typing import Union
import pandas as pd
from pathlib import Path


def parse_excel_hparams(
    file_path: Path = Path(__file__).parent / "hyperparameters.xlsx",
    sheet_name: Union[str, int] = 0,
    remove_distil=True,
    remove_global=True,
):
    assert file_path.exists()
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=1, engine="openpyxl")

    df = df.dropna(subset=["EXP ID"])  # Filter lines without values

    # "Unnamed: 0" column has a simple identifier per method.
    # B1: Magnitude Local
    # B2: Magnitude Global
    # B3: Magnitude Local W/ Distillation
    # B4: Magnitude Global W/ Distillation
    # C2: L0 Regu
    # C4: L0 Regu W/ Distillation
    # D1: Movement Local
    # D2: Movement Global
    # D3: Movement Local W/ Distillation
    # D4: Movement Global W/ Distillation
    # E2: Soft Movement
    # E4: Soft Movement W/ Distilation

    if remove_distil:
        df = df[~df["Unnamed: 0"].isin(["B3", "B4", "C4", "D3", "D4", "E4"])]

    if remove_global:
        df = df[~df["Unnamed: 0"].isin(["B2", "B4", "D2", "D4"])]

    # Drop Columns with no hparams
    df = df.drop(columns=["Unnamed: 0", "Unnamed: 1", "Unnamed: 7"])
    return df


if __name__ == "__main__":
    file_path = Path(__file__).parent / "movement-pruning-paper-hparams.xlsx"
    parse_excel_hparams(file_path, "Details - SQuAD")
