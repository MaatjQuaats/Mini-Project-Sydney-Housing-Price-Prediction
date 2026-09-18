from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "property_price_model.joblib"
ORIGINAL_PATH = ROOT / "datasets" / "real_estate_properties_data_cleaned.csv"
NEW_TEST_PATH = ROOT / "datasets" / "real_estate_properties_final_testing_cleaned.csv"
BEST_HYPERPARAMETERS = joblib.load("./trained_model/random_forest_reg_model.joblib")["hyperparameters"]
FEATURES = [
    "num_bath",
    "num_bed",
    "num_parking",
    "property_size",
    "day",
    "month",
    "year",
    "blacktown",
    "mosman",
    "parramatta",
    "Apartment",
    "Block of units",
    "Duplex/semi-detached",
    "House",
    "Townhouse",
    "Unit"
]

def make_training_data(original_path, new_test_path):
    try:
        original_data = pd.read_csv(original_path)
        new_test_data = pd.read_csv(new_test_path)
    except:
        raise FileNotFoundError("Dataset(s) not found.")

    required = ["price"] + FEATURES

    for name, data in [
        ("Original dataset", original_data),
        ("New testing dataset", new_test_data)
    ]:
        missing = [column for column in required if column not in data.columns]

        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")

    data = pd.concat(
        [original_data[required], new_test_data[required]],
        axis = 0,
        ignore_index = True
    )

    if data.isna().any().any():
        raise ValueError("The combined dataset contains missing values.")

    X = data[FEATURES]
    y = data["price"]

    return X, y


def train(original_path, new_test_path, best_parameters):
    try:
        X, y = make_training_data(original_path, new_test_path)
    except FileNotFoundError as error:
        print(f"Type: {type(error)}")
        print(f"Message: {str(error)}")
        return
    except ValueError as error:
        print(f"Type: {type(error)}")
        print(f"Message: {str(error)}")
        return
    else:
        print("Data loaded successfully.")
    finally:
        print("Finish the attempt.")

    parameters = {
        "random_state": 42,
        "n_jobs": -1,
        **best_parameters
    }

    model = RandomForestRegressor(**parameters)
    model.fit(X, y)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "hyperparameters": model.get_params()
        },
        MODEL_PATH
    )

    print(f"Trained on {len(X)} properties.")
    print(f"Saved model to: {MODEL_PATH}")

    return model

if __name__ == "__main__":
    train(ORIGINAL_PATH, NEW_TEST_PATH, BEST_HYPERPARAMETERS)