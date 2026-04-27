import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Sephora Product Success Predictor",
    page_icon="SP",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #f8f5f2;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #1f1f1f;
}

p, li, label {
    color: #333333;
}

div[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #ddd2c8;
    border-radius: 14px;
    padding: 0.9rem;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.04);
}

.stButton > button {
    background-color: #2f2a28 !important;
    color: #ffffff !important;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}

.stButton > button p {
    color: #ffffff !important;
}

.stButton > button:hover {
    background-color: #5a4638 !important;
    color: #ffffff !important;
}

.stButton > button:hover p {
    color: #ffffff !important;
}

hr {
    border-top: 1px solid #ddd2c8;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with open("final_sephora_model.pkl", "rb") as f:
        return cloudpickle.load(f)


model = load_model()


def clean_text(text):
    import re
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_input_df(review_text, ingredients, highlights, price_usd):
    review_text_clean = clean_text(review_text)
    ingredients_clean = clean_text(ingredients)
    highlights_clean = clean_text(highlights)

    combined_text = f"{review_text_clean} {ingredients_clean} {highlights_clean}".strip()
    ingredient_count = len([item for item in str(ingredients).split(",") if item.strip()])
    review_length = len(review_text_clean.split())

    input_df = pd.DataFrame([{
        "combined_text": combined_text,
        "price_usd": price_usd,
        "ingredient_count": ingredient_count,
        "review_length": review_length
    }])

    return input_df, ingredient_count, review_length


def get_prediction_probability(model, input_df):
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(input_df)[0][1])
    return None


def show_probability_interpretation(probability):
    if probability is None:
        st.write("The model produced a classification result, but probability scores are not available.")
    elif probability >= 0.75:
        st.success("Strong likelihood of a high-rated review. The product information strongly matches patterns linked to positive outcomes.")
    elif probability >= 0.55:
        st.info("Moderate likelihood of a high-rated review. Some positive indicators are present, but the model is not extremely confident.")
    elif probability >= 0.40:
        st.warning("Uncertain prediction. The model sees mixed signals in the product information and review text.")
    else:
        st.error("Low likelihood of a high-rated review. The input does not strongly match patterns linked to high-rated outcomes.")


def show_shap_explanation(model, input_df):
    st.subheader("Explainable AI: Prediction Drivers")

    st.write(
        "This section explains which input features influenced the model’s prediction for this specific product example."
    )

    try:
        import shap

        if hasattr(model, "named_steps"):
            final_step_name = list(model.named_steps.keys())[-1]
            final_model = model.named_steps[final_step_name]

            transformed_input = model[:-1].transform(input_df)

            feature_names = None
            try:
                feature_names = model[:-1].get_feature_names_out()
            except Exception:
                pass

            if feature_names is not None:
                transformed_input = pd.DataFrame(
                    transformed_input.toarray() if hasattr(transformed_input, "toarray") else transformed_input,
                    columns=feature_names
                )

            explainer = shap.Explainer(final_model, transformed_input)
            shap_values = explainer(transformed_input)

            fig = plt.figure(figsize=(8, 4))
            shap.plots.bar(shap_values[0], show=False, max_display=10)
            st.pyplot(fig, bbox_inches="tight")
            plt.close(fig)

        else:
            explainer = shap.Explainer(model, input_df)
            shap_values = explainer(input_df)

            fig = plt.figure(figsize=(8, 4))
            shap.plots.bar(shap_values[0], show=False, max_display=10)
            st.pyplot(fig, bbox_inches="tight")
            plt.close(fig)

    except Exception:
        st.write("A simplified explanation is shown below based on the engineered input features.")

        explanation_df = pd.DataFrame({
            "Feature": ["Price", "Ingredient Count", "Review Length"],
            "Input Value": [
                float(input_df["price_usd"].iloc[0]),
                int(input_df["ingredient_count"].iloc[0]),
                int(input_df["review_length"].iloc[0])
            ]
        })

        st.dataframe(explanation_df, use_container_width=True)


st.title("Sephora Product Success Predictor")

st.write(
    "This deployed machine learning dashboard predicts whether a Sephora product review is likely to be high-rated "
    "using review text, product details, price, ingredient count, and review length."
)

st.caption("Portfolio focus: machine learning deployment, explainable AI, and retail product strategy.")

st.divider()

overview_col1, overview_col2, overview_col3 = st.columns(3)

with overview_col1:
    st.metric("Model Goal", "High-Rated Review")

with overview_col2:
    st.metric("Input Type", "Text + Numeric")

with overview_col3:
    st.metric("Use Case", "Retail Insights")

st.divider()

left_col, right_col = st.columns([1.2, 0.8])

with left_col:
    st.subheader("Make a Prediction")

    review_text = st.text_area(
        "Review Text",
        placeholder="Example: Great hydrating serum that leaves skin glowing and smooth.",
        height=140
    )

    ingredients = st.text_area(
        "Ingredients",
        placeholder="Example: water, glycerin, niacinamide, vitamin c",
        height=120
    )

    highlights = st.text_area(
        "Highlights",
        placeholder="Example: brightening, hydrating, vegan",
        height=100
    )

    price_usd = st.number_input(
        "Price (USD)",
        min_value=0.0,
        value=25.0,
        step=1.0
    )

    st.caption("Tip: More detailed product information gives the model stronger signals.")

    predict_clicked = st.button("Predict Product Review Outcome", use_container_width=True)

with right_col:
    st.subheader("Model Summary")

    st.write(
        "The model identifies patterns in review language and product-related features that are associated with "
        "higher-rated Sephora products."
    )

    st.write("**Inputs used:**")
    st.write("- Review text")
    st.write("- Ingredients")
    st.write("- Product highlights")
    st.write("- Price")
    st.write("- Ingredient count")
    st.write("- Review length")

    st.write("**Prediction output:**")
    st.write("- High-rated review")
    st.write("- Low-rated review")
    st.write("- Probability score, when available")


if predict_clicked:
    if not review_text.strip() and not ingredients.strip() and not highlights.strip():
        st.warning("Please enter review text, ingredients, or product highlights before running a prediction.")
    else:
        try:
            input_df, ingredient_count, review_length = build_input_df(
                review_text,
                ingredients,
                highlights,
                price_usd
            )

            prediction = model.predict(input_df)[0]
            probability = get_prediction_probability(model, input_df)

            st.divider()
            st.subheader("Prediction Results")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                if prediction == 1:
                    st.success("High-Rated Review")
                else:
                    st.error("Low-Rated Review")

            with result_col2:
                if probability is not None:
                    st.metric("High-Rating Probability", f"{probability:.3f}")
                else:
                    st.metric("Probability", "N/A")

            with result_col3:
                st.metric("Price", f"${price_usd:.2f}")

            feature_col1, feature_col2 = st.columns(2)

            with feature_col1:
                st.metric("Ingredient Count", ingredient_count)

            with feature_col2:
                st.metric("Review Length", review_length)

            st.subheader("Prediction Interpretation")
            show_probability_interpretation(probability)

            st.divider()
            show_shap_explanation(model, input_df)

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.write(
                "Check that final_sephora_model.pkl is uploaded to the same folder as app.py "
                "and that the model was saved with the same feature names used in this app."
            )

st.divider()

st.subheader("Model Performance")

st.write(
    "This section visually summarizes how the model performed before deployment. "
    "The chart connects the model’s evaluation results to the project goal of identifying high-rated Sephora product reviews."
)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric("Accuracy", "0.80")

with metric_col2:
    st.metric("F1 Score", "0.78")

with metric_col3:
    st.metric("ROC-AUC", "0.82")

with metric_col4:
    st.metric("Validation Strategy", "80/20 Split")

metrics = ["Accuracy", "F1 Score", "ROC-AUC"]
values = [0.80, 0.78, 0.82]

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(metrics, values)
ax.set_ylim(0, 1)
ax.set_title("Model Performance Overview")
ax.set_ylabel("Score")
ax.set_xlabel("Evaluation Metric")

for i, value in enumerate(values):
    ax.text(i, value + 0.02, f"{value:.2f}", ha="center")

st.pyplot(fig)
plt.close(fig)

st.caption(
    "The model shows consistent performance across accuracy, F1 score, and ROC-AUC. "
    "This supports the goal of predicting high-rated product reviews using both text-based and structured product features."
)

st.divider()

st.subheader("Business Impact")

st.write(
    "This dashboard shows how machine learning can support beauty retail decision-making. "
    "By predicting whether a product review pattern is likely to be high-rated, the tool can help teams understand "
    "what kinds of product descriptions, ingredient profiles, pricing patterns, and customer language are connected to stronger product outcomes."
)

impact_col1, impact_col2, impact_col3 = st.columns(3)

with impact_col1:
    st.write("**Retail Strategy**")
    st.write("- Better product strategy")
    st.write("- Smarter pricing decisions")
    st.write("- Stronger assortment planning")

with impact_col2:
    st.write("**Customer Experience**")
    st.write("- Easier product discovery")
    st.write("- Better recommendations")
    st.write("- Faster decision-making")

with impact_col3:
    st.write("**Product Analytics**")
    st.write("- Better personalization")
    st.write("- Improved shopping experience")
    st.write("- More explainable recommendations")

st.divider()

st.subheader("Limitations and Future Improvements")

st.write(
    "The model is useful as a portfolio deployment, but it should not be treated as a perfect business decision system. "
    "Future versions could include more product metadata, larger review samples, real-time Sephora data, fairness checks by price group, "
    "and stronger explainability tools."
)
