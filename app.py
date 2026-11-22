import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Store Ordering Assistant", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600&display=swap');

h1 {
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-weight: 600 !important;
    font-size: 2.6rem !important;
    letter-spacing: 0.5px;
}
div[data-testid="stMetric"] {
    background-color: #2A2A30;
    border: 1px solid rgba(224, 184, 76, 0.5);
    border-radius: 12px;
    padding: 14px 18px;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #2A2A30;
    border-radius: 12px;
}
.order-box {
    background-color: #2A2A30;
    border: 1px solid rgba(224, 184, 76, 0.7);
    border-radius: 14px;
    padding: 18px 26px;
    font-size: 1.5rem;
    line-height: 1.5;
    margin: 8px 0 18px 0;
}
.order-box .big { color: #F2C94C; font-size: 2.6rem; font-weight: 600; }
.order-box small { color: #CFC6DA; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("Demand Forecast : Walmart CA_1 (Food Items)")
st.caption("A replay of 25 Apr - 22 May 2016: the orders below are what the model would "
           "have recommended on 24 Apr, compared with what customers actually bought. "
           "Data: M5 (Walmart). Product names are anonymized, so items are described "
           "by department and typical daily sales.")

df = pd.read_csv("results/val_predictions.csv", parse_dates=["date"])

# Readable item list
info = df.groupby("item_id").agg(dept=("dept_id", "first"),
                                 avg=("pred_flat", "mean")).reset_index()
info["speed"] = pd.cut(info["avg"], bins=[-0.01, 0.5, 2, 10, 1000],
                       labels=["very slow", "slow", "medium", "fast"])
info = info.sort_values("avg", ascending=False)
info["label"] = (info["dept"] + " · " + info["speed"].astype(str) + " seller · ~"
                 + info["avg"].round(1).astype(str) + "/day · " + info["item_id"])
label_to_item = dict(zip(info["label"], info["item_id"]))

plans = {
    "Lean - order less, risk some sell-outs": "pred_l1",
    "Balanced - order about the recent average": "pred_flat",
    "Safe - avoid sell-outs, accept leftovers": "pred_q90",
}

left, right = st.columns([2, 1])
with left:
    choice = st.selectbox("1. Choose a product", list(label_to_item))
with right:
    plan = st.radio("2. How careful do you want to be?", list(plans))

item = label_to_item[choice]
col = plans[plan]

# Data for this product, grouped into weeks
d = df[df["item_id"] == item].sort_values("date").copy()
d["week"] = (d["date"] - d["date"].min()).dt.days // 7
weekly = d.groupby("week").agg(start=("date", "min"),
                               sold=("sales", "sum"),
                               plan_units=(col, "sum")).reset_index()
weekly["Week"] = ("Week " + (weekly["week"] + 1).astype(str) + " · from "
                  + weekly["start"].dt.strftime("%d %b"))
weekly["Order"] = np.ceil(weekly["plan_units"]).astype(int)

total_order = int(weekly["Order"].sum())
per_week = int(round(total_order / len(weekly)))

# Big answer box
st.markdown(
    f"<div class='order-box'>Order <span class='big'>{total_order:,}</span> units "
    f"over the next 4 weeks<br><small>That is about {per_week:,} units per week. "
    f"Plan: {plan.split(' - ')[0]}</small></div>",
    unsafe_allow_html=True)

# Chart and weekly order list
chart_col, table_col = st.columns([2, 1])
with chart_col:
    with st.container(border=True):
        chart_df = (weekly.set_index("Week")[["sold", "Order"]]
                    .rename(columns={"sold": "What customers actually bought",
                                     "Order": "What you would order"}))
        st.bar_chart(chart_df, color=["#E6DAF2", "#F2C94C"], height=360)
with table_col:
    st.markdown("**Weekly order list**")
    st.dataframe(weekly[["Week", "Order"]].rename(columns={"Order": "Units to order"}),
                    hide_index=True, width="stretch")

# How it would have turned out
demand = d["sales"].sum()
stock = d[col].sum()
short = (d["sales"] - d[col]).clip(lower=0).sum()
leftover = (d[col] - d["sales"]).clip(lower=0).sum()
served = demand - short

if demand > 0:
    st.info(f"With this plan, you would have been able to serve about "
            f"{served / demand:.0%} of customer demand ({served:.0f} of {demand:.0f} units) "
            f"and had about {leftover:.0f} units left over. "
            f"(Counted day by day: stock is not carried to the next day.)")
else:
    st.info("This product had no sales in this period.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers wanted", f"{demand:.0f}")
c2.metric("Units you would order", f"{stock:.0f}")
c3.metric("Customers you could serve", f"{served / demand:.0%}" if demand > 0 else "n/a")
c4.metric("Left over, unsold", f"{leftover:.0f}")

# Day-by-day detail
with st.expander("See it day by day"):
    daily = (d.set_index("date")[["sales", col]]
             .rename(columns={"sales": "What customers actually bought",
                              col: "Planned stock for the day"}))
    st.line_chart(daily, color=["#E6DAF2", "#F2C94C"], height=320)
