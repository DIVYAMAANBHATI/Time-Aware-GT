import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Training Comparison: Epoch vs Validation")

# Load CSVs directly (no upload needed)
gcn = pd.read_csv("gcn_training_log.csv")
gt = pd.read_csv("gt_training_log.csv")
tagt = pd.read_csv("tagt_training_log.csv")

# Plot
plt.figure()

plt.plot(gcn["Epoch"], gcn["Val"], label="GCN")
plt.plot(gt["Epoch"], gt["Val_Acc"], label="GT")

# Adjust if TAGT column name differs
tagt_val_col = [col for col in tagt.columns if "val" in col.lower()][0]
plt.plot(tagt["epoch"], tagt[tagt_val_col], label="TAGT")

plt.xlabel("Epoch")
plt.ylabel("Validation Metric")
plt.title("Epoch vs Validation Comparison")
plt.legend()

st.pyplot(plt)

# Best values table
results = pd.DataFrame({
    "Model": ["GCN", "GT", "TAGT"],
    "Best Val": [
        gcn["Val"].max(),
        gt["Val_Acc"].max(),
        tagt[tagt_val_col].max()
    ]
})

st.subheader("Best Validation Accuracy")
st.table(results)