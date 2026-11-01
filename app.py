import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance Coach", page_icon="🎓")
b = joblib.load("models/advice_model.pkl")
model, features, hours = b["model"], b["features"], b["hour_features"]
ranges, target = b["ranges"], b["target"]
pretty = lambda c: c.replace("_", " ").replace(" Per Day", "").strip()

st.title("🎓 Student Performance Coach")
st.caption(f"Predicts {pretty(target)} from your daily routine and suggests time trade-offs "
           f"that are associated with a higher score. Typical model error: ±{b['test_MAE']:.2f}.")

st.sidebar.header("Your typical day (hours)")
s = dict(b["defaults"])
for c in hours:
    lo, hi = ranges[c]
    s[c] = st.sidebar.slider(pretty(c), lo, hi, float(min(max(s[c], lo), hi)), step=0.5)
total = sum(s[c] for c in hours)
st.sidebar.write(f"Total planned: **{total:.1f} h** of 24")
if total > 24:
    st.sidebar.warning("That adds up to more than 24 hours.")


def predict(row):
    return float(model.predict(pd.DataFrame([row])[features])[0])


base = predict(s)
st.metric(f"Predicted {pretty(target)} (dataset range {b['y_range'][0]:.1f}-{b['y_range'][1]:.1f})",
          f"{base:.2f}")

# Trade-offs only: move 1 hour between activities. Never takes time from sleep,
# and never leaves the range of the training data.
sleep = [c for c in hours if "sleep" in c.lower()]
tips = []
for src in hours:
    for dst in hours:
        if src == dst or src in sleep or s[src] < ranges[src][0] + 1:
            continue
        if s[dst] + 1 > ranges[dst][1]:
            continue
        alt = dict(s)
        alt[src] -= 1
        alt[dst] += 1
        tips.append((f"Move 1 hour from **{pretty(src)}** to **{pretty(dst)}**", predict(alt) - base))
tips.sort(key=lambda t: t[1], reverse=True)

st.subheader("Best time trade-offs for you")
good = [t for t in tips if t[1] > 0.005][:3]
if good:
    for label, gain in good:
        st.write(f"{label} → about **+{gain:.3f}** predicted points")
else:
    st.write("No one-hour swap improves the prediction. Try moving a slider, or you may already be at the edge of what the data covers.")

st.subheader("What matters most overall")
try:
    imp = pd.read_csv("models/importance.csv").head(8)
    imp["feature"] = imp["feature"].map(pretty)
    st.bar_chart(imp.set_index("feature")["importance"])
except FileNotFoundError:
    pass

st.info("This model finds patterns in survey data from students, mainly in India and South Asia. "
        "It shows associations, not guaranteed cause and effect, and small differences may be noise.")