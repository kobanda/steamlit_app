import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Refresher Training on Anthropometric Measures", layout="wide")

# Global Custom Style
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #f0f9f4;
            padding: 12px;
            border-radius: 8px;
            margin-right: 8px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2E8B57 !important;
            color: white !important;
        }
        .report-container {
            background-color: #f9fdfb;
            padding: 2rem;
            border-radius: 10px;
            border: 1px solid #d0e8dc;
        }
        .report-container h3 {
            color: #2E8B57;
        }
        .report-container hr {
            border: none;
            border-top: 1px solid #d0e8dc;
            margin: 20px 0;
        }
        .report-container table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        .report-container th, .report-container td {
            border: 1px solid #ddd;
            padding: 10px;
        }
        .report-container th {
            background-color: #eafaf1;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

# Main Title
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #2E8B57;'>Refresher Training on Anthropometric Measures</h1>
        <h4 style='color: #555;'>Assessing Weight, Length, MUAC, and Head Circumference on Measurer Variability</h4>
    </div>
    <hr style='border: 1px solid #2E8B57;'>
""", unsafe_allow_html=True)

# Sidebar Upload
st.sidebar.header("Step 1: Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    expected_cols = ['infant_id', 'Weight', 'Length', 'MUAC', 'hc', 'Round', 'RA']
    if not all(col in df.columns for col in expected_cols):
        st.error("Uploaded file is missing required columns.")
        st.stop()

    if df['Weight'].mean() > 100:
        df['Weight'] = round(df['Weight'] / 1000,4)

    st.sidebar.header("Step 2: Filter by RA")
    ras = sorted(df['RA'].unique())
    selected_ras = st.sidebar.multiselect("Select RA(s)", ras, default=ras)
    filtered_df = df[df['RA'].isin(selected_ras)]

    tab1, tab2, tab3, tab4,tab5= st.tabs([
        "Raw Data", "Intra-Measurer Analysis", "Inter-Measurer ANOVA", "Summary Report","JKK Vs RAs Measurements"
    ])

    with tab1:
        st.subheader("Raw Data View")
        st.dataframe(filtered_df, use_container_width=True)

    with tab2:
        st.subheader("Intra-Measurer Variability")
        intra_results = []
        for anthro in ['Weight', 'Length', 'MUAC', 'hc']:
            for ra in filtered_df['RA'].unique():
                subset = filtered_df[filtered_df['RA'] == ra]
                child_vars = []
                for child in subset['infant_id'].unique():
                    values = subset[subset['infant_id'] == child][anthro].values
                    if len(values) == 2:
                        child_vars.append(np.var(values, ddof=1))
                if child_vars:
                    avg_var = np.mean(child_vars)
                    intra_results.append({
                        'Anthropometry': anthro,
                        'RA': ra,
                        'Avg Variance': round(avg_var, 4)
                    })
        intra_df = pd.DataFrame(intra_results)
        st.dataframe(
            intra_df.style
                .background_gradient(cmap="YlGnBu", subset=["Avg Variance"])
                .format({"Avg Variance": "{:.4f}"}),
            use_container_width=True
        )

        def evaluate_performance(value):
            if value < 0.01:
                return "Excellent"
            elif value < 0.05:
                return "Needs Improvement"
            else:
                return "Review Needed"

        evaluation_df = intra_df.copy()
        evaluation_df['Assessment'] = evaluation_df['Avg Variance'].apply(evaluate_performance)

        st.subheader("Performance Assessment")
        st.dataframe(evaluation_df.style.applymap(
            lambda x: "color: green" if x == "Excellent"
            else ("color: orange" if x == "Needs Improvement" else "color: red"),
            subset=['Assessment']), use_container_width=True)

        st.subheader("Intra-Measurer Variance Chart")
        selected_anthro = st.selectbox("Choose Measurement to Plot", ['Weight', 'Length', 'MUAC', 'hc'])
        chart_df = intra_df[intra_df['Anthropometry'] == selected_anthro]
        fig = px.bar(chart_df, x='RA', y='Avg Variance', color='Avg Variance',
                     color_continuous_scale='RdYlGn_r',
                     title=f'{selected_anthro} Variance by RA')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Inter-Measurer Variability (ANOVA)")
        anova_results = []
        for anthro in ['Weight', 'Length', 'MUAC', 'hc']:
            groups = [filtered_df[filtered_df['RA'] == ra][anthro].dropna().values
                      for ra in filtered_df['RA'].unique()]
            if all(len(g) > 1 for g in groups):
                f_stat, p_val = f_oneway(*groups)
                anova_results.append({
                    'Anthropometry': anthro,
                    'F-statistic': round(f_stat, 3),
                    'p-value': round(p_val, 4),
                    'Significant Difference': "Yes" if p_val < 0.05 else "No"
                })

        anova_df = pd.DataFrame(anova_results)
        st.dataframe(anova_df.style.applymap(
            lambda x: "color: red" if x == "Yes" else "color: green",
            subset=['Significant Difference']), use_container_width=True)
        st.markdown("### Interpretation of ANOVA Results")
        with st.expander("What does this table mean? Click to understand"):
            st.markdown("""
            - This table compares how different RAs measured the same Anthro across children.
            - If the **p-value is less than 0.05**, it means the differences in measurements between RAs are **statistically significant**.
            - In simple terms:
                - **Yes** under "Significant Difference" means **RAs are measuring differently**, and we should investigate.
                - **No** means their measurements are **mostly aligned**, which is good!
            - The **F-statistic** is a technical value used for comparison, higher values with low p-values usually mean bigger differences.
            """)
    with tab4:
        st.markdown("<div class='report-container'>", unsafe_allow_html=True)

        st.markdown("## Anthropometric Standardization Report")
        st.markdown("This report summarizes intra- and inter-Measurer variability for weight, length, MUAC, and head circumference. Consistency across and within Measurers is key for data reliability.")
        st.markdown("---")

        sections = {
            "Weight": {
                "intra": {"MMK": "0.0023", "LAM": "**0.0007**", "MB": "0.0019", "SK": "0.0018"},
                "f_stat": "0.001", "p_val": "0.9999",
                "notes": [
                    "All RAs showed extremely low intra-Measurer variance—excellent repeatability.",
                    "**LAM had the lowest variance**, suggesting very precise technique and scale use."
                ],
                "recommend": [
                    "Maintain current practices",
                    "Calibrate the scales periodically",
                    "Each Measurer must do there individual work"
                ]
            },
            "Length": {
                "intra": {"MMK": "**0.0856**", "LAM": "0.4563", "MB": "0.3906", "SK": "0.1981"},
                "f_stat": "0.127", "p_val": "0.9440",
                "notes": [
                    "**MMK** showed highest precision",
                    "**LAM and MB** had higher variability—likely due to difficulty with infant movement or reading board"
                ],
                "recommend": [
                    "LAM and MB should check on positioning, reading the scale",
                    "Engage the mother to help calm down the baby",
                    "Exercise Patience with the child"
                ]
            },
            "MUAC": {
                "intra": {"MMK": "0.0600", "LAM": "0.0487", "MB": "**0.1600**", "SK": "**0.0394**"},
                "f_stat": "0.116", "p_val": "0.9502",
                "notes": [
                    "**SK** had best performance; **MB** showed high intra-Measurer variance",
                    "Issues likely with tape tension, positioning, or child movement"
                ],
                "recommend": [
                    "MB please check on tape positioning",
                    "Ensure elbow relaxed, read at eye-level",
                    "Ensure to place the tape at the correct location"
                ]
            },
            "Head Circumference": {
                "intra": {"MMK": "0.0231", "LAM": "0.0394", "MB": "**0.1644**", "SK": "0.0363"},
                "f_stat": "0.178", "p_val": "0.9108",
                "notes": [
                    "**MMK** was most consistent. MB showed highest variance",
                    "Problem likely due to inconsistent tape positioning"
                ],
                "recommend": [
                    "MB to practice snug placement and reading",
                    "Try supervised peer assessments"
                ]
            }
        }

        for section, content in sections.items():
            st.markdown(f"### {section}")
            st.markdown("#### Intra-Measurer Variability")
            st.markdown(pd.DataFrame.from_dict(content["intra"], orient='index', columns=["Avg Variance"]).to_html(), unsafe_allow_html=True)
            st.markdown("#### Interpretation")
            for line in content["notes"]:
                st.markdown(f"- {line}")
            st.markdown("#### Inter-Measurer Variability")
            st.markdown(f"| F-stat | p-value |\n|--------|---------|\n| {content['f_stat']} | {content['p_val']} |")
            st.markdown("#### Recommendations")
            for rec in content["recommend"]:
                st.markdown(f"- {rec}")
            st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("### Summary Table")
        st.markdown("""
        | Metric | Best RA (Precision) | Needs Support |
        |--------|---------------------|---------------|
        | Weight | LAM                 | None          |
        | Length | MMK                 | LAM, MB       |
        | MUAC   | SK                  | MB            |
        | hc     | MMK                 | MB            |
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
    #tab5 = st.tabs(["JKK R1 Comparison"])[0]

    with tab5:
        st.subheader("Compare Each RA’s Measurements Against JKK Measurements")

        # Filter Round 2 data only
        round2_df = df[df['Round'] == 'R2'].copy()

        # Get JKK's R2 data
        jkk_df = round2_df[round2_df['RA'] == 'JKK'].copy()

        if jkk_df.empty:
            st.warning("No measurements found for JKK.")
        else:
            comparison_results = []
            other_ras = round2_df['RA'].unique()
            other_ras = [ra for ra in other_ras if ra != 'JKK']

            for ra in other_ras:
                ra_df = round2_df[round2_df['RA'] == ra].copy()

                # Merge JKK and current RA on infant_id
                merged = pd.merge(
                    jkk_df,
                    ra_df,
                    on='infant_id',
                    suffixes=('_JKK', f'_{ra}')
                )

                if merged.empty:
                    continue

                for metric in ['Weight', 'Length', 'MUAC', 'hc']:
                    jkk_vals = merged[f'{metric}_JKK']
                    ra_vals = merged[f'{metric}_{ra}']
                    valid = jkk_vals.notna() & ra_vals.notna()

                    if valid.sum() > 0:
                        diffs = np.abs(jkk_vals[valid] - ra_vals[valid])
                        mad = diffs.mean()
                        comparison_results.append({
                            'RA': ra,
                            'Anthropometry': metric,
                            'Mean Absolute Difference': round(mad, 3),
                            'Compared N': int(valid.sum())
                        })

            if comparison_results:
                comparison_df = pd.DataFrame(comparison_results)

                st.dataframe(
                    comparison_df.pivot(index='RA', columns='Anthropometry', values='Mean Absolute Difference')
                    .style.background_gradient(cmap='RdYlGn_r', axis=1),
                    use_container_width=True
                )

                st.subheader("Visual Comparison with JKK Measurements")

                metric_to_plot = st.selectbox("Select Measurement", ['Weight', 'Length', 'MUAC', 'hc'])

                plot_df = comparison_df[comparison_df['Anthropometry'] == metric_to_plot]
                fig = px.bar(plot_df, x='RA', y='Mean Absolute Difference',
                             color='Mean Absolute Difference', color_continuous_scale='RdYlGn_r',
                             title=f"Mean Absolute Difference vs JKK for {metric_to_plot} Measurements")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                - Lower bars = closer agreement with JKK.
                - Only infants measured by both JKK and the RA are used.
                - Missing values are excluded from analysis.
                """)
            else:
                st.info("No overlapping data between JKK and other RAs.")

else:
    st.info("Please upload a CSV file with anthropometric test data to begin.")




