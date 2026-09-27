import pandas as pd
import json

df = pd.read_csv("test_trainer/mock_data 2b.csv")
cols = [
    "operator_type_percentage",
    "food_type_percentage",
    "sales_per_operator_type",
    "sales_per_food_type",
]
df_list = [
    df.drop(
        columns=cols,
        axis=1,
    )
]
for col in [
    "operator_type_percentage",
    "food_type_percentage",
    "sales_per_operator_type",
    "sales_per_food_type",
]:
    col_list = []
    for row in df[col]:
        _dict = json.loads(row)
        col_list.append({f"operator_type_percentage_{k}": v for k, v in _dict.items()})
    df_col = pd.DataFrame.from_dict(col_list)
    df_list.append(df_col)
df_cols = pd.concat(df_list, axis=1)
df_cols.to_csv("test_trainer/mock_data 2b_cols.csv")
