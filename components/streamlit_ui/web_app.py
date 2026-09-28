import pandas as pd
import argparse
import streamlit as st
import wandb
import mlflow
import yaml
import requests

cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
               'Income_Category', 'Card_Category']

CONFIG_FILE = '../../config.yaml'
with open(CONFIG_FILE) as f:
    api_url = yaml.safe_load(f)['main']['api_url']


def go(args):
    st.set_page_config(
        page_title="Customer Churn Prediction",
        layout="centered"
    )

    st.title("Customer Churn Prediction:")

    # Load test data from wandb
    run = wandb.init(project="CustomerChurnAI", job_type="streamlit_ui")
    run.config.update(args)

    test_data_path = run.use_artifact(args.test_artifact).file()
    df = pd.read_csv(test_data_path)

    # Load export model from wandb
    model_local_path = run.use_artifact(args.export_model).download()
    model = mlflow.sklearn.load_model(model_local_path)

    if df is not None:
        st.success(f"Dataset loaded: {len(df)} customers")

        # Select customer
        customer_number = st.number_input(
            "Select Customer",
            min_value=1,
            max_value=len(df),
            value=1,
            step=1
        )

        # Python index starts from 0
        customer = df.iloc[customer_number - 1]

        st.divider()

        st.subheader(f"Customer #{customer_number}")
        st.subheader("Customer Overview")

        col1, col2 = st.columns(2)

        # Left column
        with col1:

            if "Customer_Age" in df.columns:
                st.metric(
                    "Age",
                    customer["Customer_Age"]
                )

            if "Dependent_count" in df.columns:
                st.metric(
                    "Dependents",
                    customer["Dependent_count"]
                )

            if "Months_on_book" in df.columns:
                st.metric(
                    "Account Age",
                    f"{customer['Months_on_book']} months"
                )

            if "Months_Inactive_12_mon" in df.columns:
                st.metric(
                    "Months Inactive",
                    customer["Months_Inactive_12_mon"]
                )

        # Right column
        with col2:

            if "Credit_Limit" in df.columns:
                st.metric(
                    "Credit Limit",
                    f"${customer['Credit_Limit']:,.2f}"
                )

            if "Total_Trans_Amt" in df.columns:
                st.metric(
                    "Transaction Amount",
                    f"${customer['Total_Trans_Amt']:,.2f}"
                )

            if "Total_Trans_Ct" in df.columns:
                st.metric(
                    "Transactions",
                    customer["Total_Trans_Ct"]
                )

            if "Avg_Utilization_Ratio" in df.columns:
                st.metric(
                    "Utilization Ratio",
                    f"{customer['Avg_Utilization_Ratio']:.1%}"
                )

        # Optional: inspect all features
        with st.expander("View all customer features"):

            st.dataframe(
                customer.to_frame(name="Value"),
                use_container_width=True
            )



        if st.button(
            "Analyze Customer",
            type="primary",
            use_container_width=True
        ):
            st.divider()
            st.header("Churn Prediction")

            # Drop categorical columns & prepare features
            customer.drop(labels=cat_columns, inplace=True)
            X = customer.drop(labels=["Churn"]).to_frame().T

            # Model prediction
            response = requests.post(f'{api_url}/predict', json=X.iloc[0].to_dict()).json()

            predicted_value = response['prediction']
            churn_probability = response['churn_probability']

            # Actual value
            actual_value = int(customer["Churn"])

            # Labels
            predicted_label = (
                "Likely to Churn"
                if predicted_value == 1
                else "Likely to Stay"
            )

            actual_label = (
                "Churned"
                if actual_value == 1
                else "Stayed"
            )

            # Risk
            st.subheader("Risk of Churn")

            st.progress(float(churn_probability))

            st.write(f"**{churn_probability:.0%}**")

            # Results
            st.write(f"**Prediction:** {predicted_label}")
            st.write(f"**Predicted Value:** {predicted_value}")
            st.write(f"**Actual Value:** {actual_value} ({actual_label})")
    
    else:
    
        st.info(
            "Sample data was not uploaded to select a customer."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="launching the web app")

    parser.add_argument(
        "--test_artifact",
        type=str,
        help="Test artifact to be uploaded in the web app for customer selection",
        required=True
    )
    
    parser.add_argument(
        "--export_model",
        type=str,
        help="this is the export model to be used in the web app",
        required=True
    )

    args = parser.parse_args()
    go(args)