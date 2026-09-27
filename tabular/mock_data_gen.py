# Completing the generation of 100 rows of data
import os
import random
import pandas as pd
import yaml
from collections import Counter

ROW_COUNT = 60
# Extended list of operator types
operator_types = ["supermarket", "fine_dining", "bar_and_grill"]

# Source spreadsheet path lives in mock_data_config.yaml (git-ignored).
# Copy mock_data_config.yaml.example to get started.
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_data_config.yaml")
with open(config_path, "r") as f:
    mock_config = yaml.safe_load(f)

excel_df = pd.read_excel(mock_config["source_xlsx"])
excel_df = excel_df[
    excel_df["product_category_description"].isin(
        ["APPETIZERS AND COATED VEGETABLES", "POTATOES, FZN", "VEGETABLES, FZN"]
    )
]
print(excel_df.head())

# Extended list of food types
food_types = [
    "american",
    "italian",
    "asian",
    "mexican",
]

import json


def calculate_percentages(my_list):
    # Count the frequency of each element
    count = Counter(my_list)

    # Total number of elements in the list
    total_elements = len(my_list)

    # Dictionary to hold percentages
    percentages = {}

    # Calculate percentage for each element
    for element, frequency in count.items():
        percentages[element] = frequency / total_elements

    return percentages


data_rows = []
product_list = []
for i in range(ROW_COUNT):
    msa_name = random.choice(["San Diego", "Phoenix", "Portland", "Denver"])
    month = f"2023-{random.randint(1, 12)}"
    # getting ith row of excel data
    excel_row = excel_df.sample(1).to_dict(orient="records")[0]
    product_category = excel_row["product_category_description"]
    product_desc = excel_row["description"]
    product_list.append(product_desc)
    product_id = 1 + product_list.index(product_desc)
    new_item = random.choice([1, 0])
    void = random.choice([1, 0])
    recommended_product = random.choice([1, 0])
    gap = random.choice([1, 0])
    at_risk = random.choice([1, 0])
    recommended_cases = random.randint(150, 350)
    total_revenue_opportunity = random.uniform(4000, 25000)
    write_down_amount = random.randint(0, 45)
    profit_margin = random.randint(50, 100)
    total_operators = random.randint(10, 500)
    total_sales_volume = random.uniform(50000.0, 5000000.0)
    operator_type_perc_of_msa = calculate_percentages(
        [random.choice(operator_types) for _ in range(total_operators)]
    )
    food_type_perc_of_msa = calculate_percentages(
        [random.choice(food_types) for _ in range(total_operators)]
    )
    sales_per_operator_type = {
        k: total_sales_volume * val for k, val in operator_type_perc_of_msa.items()
    }
    sales_per_food_type = {
        k: total_sales_volume * val for k, val in food_type_perc_of_msa.items()
    }
    avg_annual_revenue_per_operator = random.uniform(100000.0, 1000000.0)
    operator_concentration_index = random.uniform(0.1, 1.0)
    expected_sales_volume = total_sales_volume * random.uniform(0.8, 1.2)
    sales_volume_gap = total_sales_volume - expected_sales_volume
    is_private_label = excel_row["is_exclusive_brand"]
    print(is_private_label)
    private_label_coverage = random.uniform(10.0, 90.0)
    void_space_percentage = random.uniform(5.0, 20.0) if void == 1 else 0
    total_promotion_spend = random.uniform(10000.0, 100000.0)
    sales_team_visits = random.randint(50, 300)
    supplier_private_label_spend = random.uniform(5000.0, 50000.0)
    supplier_non_private_label_spend = random.uniform(5000.0, 50000.0)
    supplier_assortment_count = random.randint(5, 20)
    private_label_product_count = (
        random.randint(10, 100) if is_private_label == 1 else 0
    )
    private_label_gap_count = random.randint(0, 10) if is_private_label == 1 else 0
    private_label_gap_value = (
        random.uniform(1000.0, 10000.0) if is_private_label == 1 else 0
    )
    impact_of_eliminating_private_label = (
        random.uniform(-100000.0, 100000.0) if is_private_label == 1 else 0
    )
    perfect_assortment_opportunity_value = random.uniform(10000.0, 50000.0)
    financial_worth_of_void_space = (
        random.uniform(-5000.0, -10000.0) if void == 1 else 0
    )
    market_opportunity_for_void_space = (
        random.uniform(5000.0, 10000.0) if void == 1 else 0
    )
    new_item_financial_worth = random.uniform(1000.0, 10000.0) if new_item == 1 else 0
    product_profitability = random.uniform(10.0, 100.0)
    sales_attempt_count = random.randint(100, 500)
    successful_sales_attempts = int(sales_attempt_count * random.uniform(0.1, 0.5))
    success_rate_per_sales_attempt = successful_sales_attempts / sales_attempt_count
    avg_revenue_per_successful_sale = random.uniform(500.0, 2000.0)
    avg_revenue_per_sales_attempt = random.uniform(200.0, 800.0)
    market_penetration_index = random.uniform(0.1, 1.0)
    competitive_intensity_index = random.uniform(0.1, 1.0)
    sales_cycle_time = random.uniform(30.0, 90.0)
    customer_retention_rate = random.uniform(10.0, 90.0)
    customer_acquisition_cost = random.uniform(100.0, 500.0)
    roi_per_sales_attempt = random.uniform(0.1, 5.0)
    cost_per_sales_attempt = random.uniform(100.0, 500.0)

    cols = [
        msa_name,
        product_category,
        product_id,
        product_desc,
        new_item,
        is_private_label,
        void,
        recommended_product,
        gap,
        at_risk,
        recommended_cases,
        total_revenue_opportunity,
        write_down_amount,
        profit_margin,
        json.dumps(operator_type_perc_of_msa),
        json.dumps(food_type_perc_of_msa),
        json.dumps(sales_per_operator_type),
        json.dumps(sales_per_food_type),
        total_operators,
        avg_annual_revenue_per_operator,
        operator_concentration_index,
        total_sales_volume,
        expected_sales_volume,
        sales_volume_gap,
        private_label_coverage,
        void_space_percentage,
        total_promotion_spend,
        supplier_private_label_spend,
        supplier_non_private_label_spend,
        supplier_assortment_count,
        private_label_product_count,
        private_label_gap_count,
        private_label_gap_value,
        impact_of_eliminating_private_label,
        perfect_assortment_opportunity_value,
        financial_worth_of_void_space,
        market_opportunity_for_void_space,
        new_item_financial_worth,
        product_profitability,
        sales_team_visits,
        successful_sales_attempts,
        success_rate_per_sales_attempt,
        avg_revenue_per_successful_sale,
        avg_revenue_per_sales_attempt,
        market_penetration_index,
        competitive_intensity_index,
        sales_cycle_time,
        customer_retention_rate,
        customer_acquisition_cost,
        roi_per_sales_attempt,
        cost_per_sales_attempt,
    ]

    data_rows.append(cols)

csv_headers = [
    "msa_name",
    "product_category",
    "product_id",
    "product_desc",
    "new_item",
    "is_private_label",
    "void",
    "recommended_product",
    "gap",
    "at_risk",
    "recommended_cases",
    "total_revenue_opportunity",
    "write_down_amount",
    "profit_margin",
    "operator_type_perc_of_msa",
    "food_type_perc_of_msa",
    "sales_per_operator_type",
    "sales_per_food_type",
    "total_operators",
    "avg_annual_revenue_per_operator",
    "operator_concentration_index",
    "total_sales_volume",
    "expected_sales_volume",
    "sales_volume_gap",
    "private_label_coverage",
    "void_space_percentage",
    "total_promotion_spend",
    "supplier_private_label_spend",
    "supplier_non_private_label_spend",
    "supplier_assortment_count",
    "private_label_product_count",
    "private_label_gap_count",
    "private_label_gap_value",
    "impact_of_eliminating_private_label",
    "perfect_assortment_opportunity_value",
    "financial_worth_of_void_space",
    "market_opportunity_for_void_space",
    "new_item_financial_worth",
    "product_profitability",
    "sales_team_visits",
    "successful_sales_attempts",
    "success_rate_per_sales_attempt",
    "avg_revenue_per_successful_sale",
    "avg_revenue_per_sales_attempt",
    "market_penetration_index",
    "competitive_intensity_index",
    "sales_cycle_time",
    "customer_retention_rate",
    "customer_acquisition_cost",
    "roi_per_sales_attempt",
    "cost_per_sales_attempt",
]
if len("csv_headers") != len(data_rows[0]):
    print(
        Exception(
            f"len(csv_headers)!= len(data_rows[0]), len(csv_headers)= {len(csv_headers)}, len(data_rows)= {len(data_rows[0])}"
        )
    )
# Create DataFrame using column headers
df = pd.DataFrame(data_rows, columns=csv_headers)
print(df.head())
cols = [
    "operator_type_perc_of_msa",
    "food_type_perc_of_msa",
    "sales_per_operator_type",
    "sales_per_food_type",
]
df_scooped = df.drop(columns=cols)

df_list = [df_scooped]

for col in cols:
    col_list = []
    # Loop through each row in the JSON column
    for row in df[col]:
        _dict = json.loads(row)
        # Create a dictionary with new column names and values
        col_list.append({f"{col}_{k}": float(v) for k, v in _dict.items()})
    # Create a DataFrame from the list of dictionaries
    df_col = pd.DataFrame(col_list)
    df_list.append(df_col)

# Concatenate the DataFrames horizontally
df_expanded = pd.concat(df_list, axis=1)
df_expanded = df_expanded.rename(
    columns={
        "total_operators": "total_operators_purchased",
        "avg_annual_revenue_per_operator": "avg_projected_annual_revenue_per_operator",
        "operator_concentration_index": "operator_concentration_index",
        "total_sales_volume": "total_actual_sales_volume",
        "expected_sales_volume": "total_sales_volume",
        "sales_team_visits": "total_sales_team_visits",
        "financial_worth_of_void_space": "financial_worth_filling_void",
        "market_opportunity_for_void_space": "total_market_opportunity_filling_void",
        "new_item_financial_worth": "predicted_financial_worth_new_item",
    }
)
df_expanded = df_expanded.round(2)
df_expanded = df_expanded.sort_values(["msa_name", "product_category", "product_id"])
df_expanded.to_csv("mock_data.csv", index=False)
print(df_expanded.head())
