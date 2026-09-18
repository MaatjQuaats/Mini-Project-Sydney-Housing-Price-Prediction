from datetime import date
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "property_price_model.joblib"
SUBURBS = ["blacktown", "mosman", "parramatta"]
TYPES = ["Apartment", "Block of units", "Duplex/semi-detached", "House", "Townhouse", "Unit"]
NUMERIC = ["num_bath", "num_bed", "num_parking", "property_size", "day", "month", "year"]
FEATURES = NUMERIC + SUBURBS + TYPES
RAW_COLUMNS = ["suburb", "type", "num_bath", "num_bed", "num_parking", "property_size", "date_sold"]


def load_model():
    artifact = joblib.load(MODEL_PATH)
    return artifact

def prepare_data(data, features):
    df = data.copy()

    df.drop(
        columns = ["price", "source_url"],
        errors = "ignore"
    )

    if df.empty:
        raise ValueError("The dataset is empty.")

    if df.columns.duplicated().any():
        raise ValueError("Column names must be unique.")

    if all(column in df.columns for column in features):
        df = df[features]

        if df.isna().any().any():
            raise ValueError("Some property information is missing.")

        if (df["property_size"] <= 0).any():
            raise ValueError("Property size must be greater than zero.")

        for columns in [SUBURBS, TYPES]:
            if not df[columns].isin([0, 1]).all().all():
                raise ValueError("Encoded categories must contain only 0 or 1.")

            if not df[columns].sum(axis = 1).eq(1).all():
                raise ValueError("Each property must have exactly one suburb and one type.")
        
        return df.astype(float)

    required = [
        "suburb",
        "type",
        "num_bath",
        "num_bed",
        "num_parking",
        "property_size",
        "date_sold"
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            "Input must contain all encoded features or raw property columns. "
            "Missing raw columns: " + ", ".join(missing)
        )

    df["date_sold"] = pd.to_datetime(
        df["date_sold"],
        format = "mixed"
    )

    df["day"] = df["date_sold"].dt.day
    df["month"] = df["date_sold"].dt.month
    df["year"] = df["date_sold"].dt.year

    df["suburb"] = df["suburb"].str.strip().str.lower()
    df["type"] = df["type"].str.strip()

    if not df["suburb"].isin(SUBURBS).all():
        raise ValueError(
            "Suburb must be blacktown, mosman or parramatta."
        )

    if not df["type"].isin(TYPES).all():
        raise ValueError(
            "Property type must be one of: " + ", ".join(TYPES)
        )

    encoder = OneHotEncoder(
        categories = [SUBURBS, TYPES],
        sparse_output = False,
        handle_unknown = "error"
    )

    encoded = encoder.fit_transform(df[["suburb", "type"]])

    df[SUBURBS + TYPES] = encoded

    X = df[features].astype(float)

    if X.isna().any().any():
        raise ValueError("Some property information is missing.")

    if (X["property_size"] <= 0).any():
        raise ValueError("Property size must be greater than zero.")

    return X


def predict(frame, artifact):
    X = prepare_data(frame, artifact["features"])
    return artifact["model"].predict(X)


def main():
    st.set_page_config(page_title = "Property Price Predictor", page_icon = "🏠", layout = "centered")
    st.title("🏠 Property Sale Price Predictor")
    st.write("Estimate a sale price in Mosman, Parramatta or Blacktown.")
    if not MODEL_PATH.exists():
        st.error("Model file missing. Place property_price_model.joblib in the same folder as app.py, then reload.")
        st.stop()
    try:
        artifact = load_model()
    except Exception as error:
        st.error(f"Cannot load the trained model: {error}")
        st.stop()

    st.caption("Prices are in Australian dollars and property size may describe land, building or total area.")
    manual, upload = st.tabs(["Enter Property Details", "Upload CSV"])
    with manual:
        with st.form("property_form"):
            left, right = st.columns(2)
            suburb = left.selectbox("Suburb", SUBURBS, format_func = str.title)
            property_type = right.selectbox("Property type", TYPES, index = 3)
            beds = left.number_input("Number of bedrooms", min_value = 0, value = 3, step = 1)
            baths = right.number_input("Number of bathrooms", min_value = 0, value = 2, step = 1)
            parking = left.number_input("Number of parking spaces", min_value = 0, value = 1, step = 1)
            size = right.number_input("Property size (m²)", min_value = 0.1, value = 500.0, step = 1.0)
            sale_date = st.date_input("Intended sale date", value = date.today())
            submitted = st.form_submit_button("Predict sale price")
        if submitted:
            row = pd.DataFrame([{"suburb": suburb, "type": property_type,
                "num_bed": beds, "num_bath": baths, "num_parking": parking,
                "property_size": size, "date_sold": sale_date.isoformat()}])
            try:
                price = float(predict(row, artifact)[0])
                st.metric("Estimated sale price", f"A${price:,.0f}")
            except Exception as error:
                st.error(f"Prediction failed: {error}")
    with upload:
        st.write("Upload one property per row.")
        st.code(",".join(RAW_COLUMNS), language = "text")
        st.caption("Upload a CSV containing unscaled property details with dates in YYYY-MM-DD format, or all 16 encoded model features.")
        st.download_button("Download blank CSV template", ",".join(RAW_COLUMNS) + "\n", "property_input_template.csv", "text/csv")
        with st.expander("CSV required format"):
            st.write(", ".join(TYPES))
            st.code(",".join(artifact["features"]), language = "text")
        uploaded = st.file_uploader("Choose your Property CSV", type = ["csv"])
        if uploaded is not None:
            try:
                frame = pd.read_csv(uploaded)
                results = frame.copy()
                results["predicted_sale_price"] = np.round(predict(frame, artifact), 2)
                st.success(f"Predicted prices for {len(results)} properties.")
                st.dataframe(results)
                st.download_button("Download predictions", results.to_csv(index = False), "property_predictions.csv", "text/csv")
            except Exception as error:
                st.error(f"CSV could not be processed: {error}")


if __name__ == "__main__":
    main()
