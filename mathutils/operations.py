import pandas as pd
import numpy as np
from typing import Union

def quantile_normalize_csv(
    input_csv: str,
    output_csv: Union[str, None] = None
) -> pd.DataFrame:
    """
    Perform quantile normalization on a matrix CSV file.

    Parameters:
    - input_csv: path to input CSV (genes in columns, sample IDs in first column)
    - output_csv: (optional) path to save the normalized matrix as CSV

    Returns:
    - pandas DataFrame: normalized matrix
    """

    # Step 1: Load CSV with sample IDs as row index
    df = pd.read_csv(input_csv, index_col=0)

    # Step 2: Sort each column
    sorted_df = pd.DataFrame(
        np.sort(df.values, axis=0),
        index=df.index,
        columns=df.columns
    )

    # Step 3: Compute mean across each row (quantile)
    mean_ranks = sorted_df.mean(axis=1).values

    # Step 4: Rank original data
    ranks = df.rank(method='min').astype(int) - 1  # 0-based index

    # Step 5: Replace ranks with quantile means
    normed = df.copy()
    for col in df.columns:
        normed[col] = ranks[col].apply(lambda r: mean_ranks[r])

    # Step 6: Save output if requested
    if output_csv:
        normed.to_csv(output_csv)

    return normed

