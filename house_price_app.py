import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .main-header h1 { font-size: 2.5rem; font-weight: 700; margin: 0; }
    .main-header p  { font-size: 1rem; opacity: 0.8; margin-top: 0.5rem; }

    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #0f3460;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .metric-card .label { font-size: 0.8rem; color: #6c757d; font-weight: 600; text-transform: uppercase; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #0f3460; }

    .predict-box {
        background: linear-gradient(135deg, #0f3460, #533483);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-top: 1rem;
    }
    .predict-box .price { font-size: 2.5rem; font-weight: 800; }
    .predict-box .label { font-size: 0.9rem; opacity: 0.8; margin-top: 0.3rem; }

    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f3460;
        border-bottom: 2px solid #0f3460;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stTabs"] button { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏠 House Price Predictor</h1>
    <p>Regression model comparison & instant price estimation</p>
</div>
""", unsafe_allow_html=True)

# ── Data generation ────────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    sqft = rng.normal(2000, 500, n).clip(500, 5000)
    beds = rng.integers(1, 7, n)
    baths = rng.integers(1, 5, n)
    year  = rng.integers(1960, 2024, n)
    lot   = rng.normal(8000, 3000, n).clip(1000, 20000)
    garage = rng.integers(0, 4, n)
    nq = rng.integers(1, 11, n)
    price = (
        150 * sqft
        + 5000 * beds
        + 8000 * baths
        - 500  * (2024 - year)
        + 2    * lot
        + 3000 * garage
        + rng.normal(0, 15000, n)
    ).clip(50000, 2000000)
    df = pd.DataFrame({
        "Square_Footage"     : sqft.round(0).astype(int),
        "Num_Bedrooms"       : beds,
        "Num_Bathrooms"      : baths,
        "Year_Built"         : year,
        "Lot_Size"           : lot.round(0).astype(int),
        "Garage_Size"        : garage,
        "Neighborhood_Quality": nq,
        "House_Price"        : price.round(-2).astype(int),
    })
    return df

df = generate_dataset()

# ── Model training ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    X = df.drop("House_Price", axis=1)
    y = df["House_Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge"            : Ridge(alpha=1.0),
        "Lasso"            : Lasso(alpha=0.1),
        "Elastic Net"      : ElasticNet(alpha=0.1),
        "KNN"              : KNeighborsRegressor(n_neighbors=5),
        "SVR"              : SVR(kernel="rbf", C=1000),
        "Decision Tree"    : DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest"    : RandomForestRegressor(n_estimators=100, random_state=42),
        "ANN"              : MLPRegressor(hidden_layer_sizes=(100,50), max_iter=500, random_state=42),
    }

    results, trained = [], {}
    for name, m in models.items():
        m.fit(Xtr, y_train)
        yp = m.predict(Xte)
        trained[name] = m
        results.append({
            "Model": name,
            "R²"   : round(r2_score(y_test, yp), 4),
            "RMSE" : round(np.sqrt(mean_squared_error(y_test, yp)), 0),
            "MAE"  : round(mean_absolute_error(y_test, yp), 0),
        })
    return trained, scaler, pd.DataFrame(results).sort_values("R²", ascending=False).reset_index(drop=True), X.columns.tolist(), X_test, y_test, scaler.transform(X_test)

trained_models, scaler, results_df, feature_cols, X_test, y_test, X_test_scaled = train_models(df)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    chosen_model = st.selectbox("Select model", list(trained_models.keys()), index=0)
    st.markdown("---")

    st.markdown("### 🔎 Enter House Features")
    sqft  = st.slider("Square Footage (ft²)",  500,  5000, 2000, 50)
    beds  = st.slider("Bedrooms",               1,    6,    3)
    baths = st.slider("Bathrooms",              1,    5,    2)
    year  = st.slider("Year Built",           1960, 2024, 2000)
    lot   = st.slider("Lot Size (ft²)",       1000, 20000, 8000, 500)
    garage = st.slider("Garage Size (cars)",   0,    4,    2)
    nq    = st.slider("Neighborhood Quality (1–10)", 1, 10, 5)

    user_input = np.array([[sqft, beds, baths, year, lot, garage, nq]])
    user_scaled = scaler.transform(user_input)
    prediction = trained_models[chosen_model].predict(user_scaled)[0]

    st.markdown(f"""
    <div class="predict-box">
        <div class="label">Estimated Price</div>
        <div class="price">${prediction:,.0f}</div>
        <div class="label">using {chosen_model}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Comparison", "🔍 Data Explorer", "📈 Visual Analysis", "🧮 Prediction Detail"])

# ── Tab 1: Model Comparison ────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Model Performance Metrics</div>', unsafe_allow_html=True)

    # Highlight best row
    def highlight_best(row):
        best_r2 = results_df["R²"].max()
        if row["R²"] == best_r2:
            return ["background-color: #d4edda; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        results_df.style.apply(highlight_best, axis=1).format({"R²": "{:.4f}", "RMSE": "${:,.0f}", "MAE": "${:,.0f}"}),
        use_container_width=True, height=370
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">R² Score (higher = better)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#0f3460" if m == chosen_model else "#8ecae6" for m in results_df["Model"]]
        ax.barh(results_df["Model"][::-1], results_df["R²"][::-1], color=colors[::-1])
        ax.axvline(1.0, color="red", linestyle="--", alpha=0.5, label="Perfect = 1.0")
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("R² Score")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-header">RMSE (lower = better)</div>', unsafe_allow_html=True)
        rmse_sorted = results_df.sort_values("RMSE")
        fig, ax = plt.subplots(figsize=(7, 4))
        colors2 = ["#0f3460" if m == chosen_model else "#ffb347" for m in rmse_sorted["Model"]]
        ax.barh(rmse_sorted["Model"][::-1], rmse_sorted["RMSE"][::-1], color=colors2[::-1])
        ax.set_xlabel("RMSE ($)")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── Tab 2: Data Explorer ───────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{len(df):,}")
    c2.metric("Features", len(feature_cols))
    c3.metric("Avg Price", f"${df['House_Price'].mean():,.0f}")
    c4.metric("Price Range", f"${df['House_Price'].min():,.0f} – ${df['House_Price'].max():,.0f}")

    st.markdown('<div class="section-header">Raw Data Sample</div>', unsafe_allow_html=True)
    st.dataframe(df.sample(10, random_state=1).reset_index(drop=True), use_container_width=True)

    st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().T.round(2), use_container_width=True)

# ── Tab 3: Visual Analysis ─────────────────────────────────────────────────────
with tab3:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="BrBG", ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown('<div class="section-header">House Price Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(df["House_Price"], kde=True, color="#0f3460", ax=ax)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}K"))
        ax.set_xlabel("House Price")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-header">Square Footage vs Price</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    scatter = ax.scatter(df["Square_Footage"], df["House_Price"],
                         c=df["Neighborhood_Quality"], cmap="viridis", alpha=0.4, s=10)
    plt.colorbar(scatter, ax=ax, label="Neighborhood Quality")
    ax.set_xlabel("Square Footage (ft²)")
    ax.set_ylabel("House Price ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="section-header">Price by Bedrooms</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        df.groupby("Num_Bedrooms")["House_Price"].median().plot(kind="bar", color="#0f3460", ax=ax)
        ax.set_xlabel("Number of Bedrooms")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
        ax.set_ylabel("Median Price")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_d:
        st.markdown('<div class="section-header">Price by Neighborhood Quality</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        df.groupby("Neighborhood_Quality")["House_Price"].median().plot(kind="bar", color="#533483", ax=ax)
        ax.set_xlabel("Neighborhood Quality (1–10)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
        ax.set_ylabel("Median Price")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── Tab 4: Prediction Detail ───────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Your House Profile</div>', unsafe_allow_html=True)

    profile_df = pd.DataFrame({
        "Feature": ["Square Footage", "Bedrooms", "Bathrooms", "Year Built", "Lot Size", "Garage Size", "Neighborhood Quality"],
        "Value"  : [f"{sqft:,} ft²", beds, baths, year, f"{lot:,} ft²", garage, f"{nq}/10"]
    })
    st.table(profile_df.set_index("Feature"))

    st.markdown('<div class="section-header">All Model Predictions for Your House</div>', unsafe_allow_html=True)
    all_preds = []
    for name, m in trained_models.items():
        p = m.predict(user_scaled)[0]
        all_preds.append({"Model": name, "Predicted Price": f"${p:,.0f}", "_val": p})

    preds_df = pd.DataFrame(all_preds).sort_values("_val", ascending=False).drop("_val", axis=1).reset_index(drop=True)

    def highlight_selected(row):
        if row["Model"] == chosen_model:
            return ["background-color: #cfe2ff; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(preds_df.style.apply(highlight_selected, axis=1), use_container_width=True)

    st.markdown('<div class="section-header">Prediction Spread Across Models</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    vals = [trained_models[m].predict(user_scaled)[0] for m in preds_df["Model"]]
    bar_colors = ["#0f3460" if m == chosen_model else "#8ecae6" for m in preds_df["Model"]]
    ax.bar(preds_df["Model"], vals, color=bar_colors)
    ax.axhline(prediction, color="red", linestyle="--", label=f"Selected: ${prediction:,.0f}")
    ax.set_ylabel("Predicted Price ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    plt.xticks(rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Built from the House Price Prediction notebook · "
    "Models: Linear Regression, Ridge, Lasso, Elastic Net, KNN, SVR, Decision Tree, Random Forest, ANN</small></center>",
    unsafe_allow_html=True
)
