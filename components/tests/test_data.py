import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data):
    """
    Verify that the dataset contains the expected columns in the correct order.

    Args:
        data: Input dataset to validate.
    """

    expected_colums = ['Customer_Age', 'Gender', 'Dependent_count',
           'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category',
           'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
           'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
           'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
           'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
           'Churn', 'Gender_Churn', 'Education_Level_Churn',
           'Marital_Status_Churn', 'Income_Category_Churn', 'Card_Category_Churn'
           ]

    these_columns = data.columns.to_numpy()

    # This also enforces the same order using numpy comparison for better performance
    assert np.array_equal(expected_colums, these_columns)


def test_row_count(data):
    """
    Verify that the number of rows in the dataset is within the expected range.

    Args:
        data: Input dataset to validate.
    """
    n_rows = data.shape[0]

    assert 8000 < n_rows < 15000


def test_similar_target_distrib(
        data: pd.DataFrame,
        ref_data: pd.DataFrame,
        kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset

    Args:
        data: Input dataset containing the new target distribution.
        ref_data: Reference dataset used for distribution comparison.
        kl_threshold: Maximum allowed KL divergence between the distributions.
    """
    dist1 = data['Churn'].value_counts().sort_index()
    dist2 = ref_data['Churn'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold
