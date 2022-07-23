import pandas as pd
import numpy as np


def stratified_sampling(df, column_name="label", split_ratio=0.9):
    """
    Splitting the dataset into groups ( here based on class labels) and \n
    then sampling data into train and test for each group based on split ratio.

     Parameters:
    ----------
    df : pandas.DataFrame
        Dataframe containing the data.
    column_name: str
        column based on which stratified sampling is done
    split_ratio : float
        Ratio with with which the dataframe is split into train and val
    Returns:
    --------
    df : pandas.DataFrame

    """
    # df = pd.read_csv(self.input_csv)
    if "stratified_status" not in list(df.columns):
        df = df.sample(frac=1).reset_index(drop=True)  # Shuffle df before assigning status
        df["stratified_status"] = ""

        for lab in df[column_name].unique():
            df_by_label = df[df[column_name] == lab].index
            train_val_count = int(split_ratio * len(df_by_label))
            df["stratified_status"][df_by_label[:train_val_count]] = "train"
            df["stratified_status"][df_by_label[train_val_count:]] = "val"

    return df
