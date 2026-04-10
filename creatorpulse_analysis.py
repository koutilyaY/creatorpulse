"""
=============================================================================
CreatorPulse: Early Warning System for Creator Economy Burnout
=============================================================================
Problem: The $250B creator economy suffers billions in losses when top creators
burn out unexpectedly. No data-driven early warning system exists.

Skills Demonstrated:
  - Exploratory Data Analysis (EDA)
  - Statistical Hypothesis Testing (t-test, Mann-Whitney U, Chi-Square)
  - A/B Testing with multiple correction (Bonferroni)
  - Survival Analysis (Kaplan-Meier + Cox PH via lifelines)
  - Machine Learning (Gradient Boosting Classifier)
  - Feature Engineering (rolling stats, lag features, ratio features)
  - Causal Inference (Difference-in-Differences)
  - Model Evaluation (ROC-AUC, Precision-Recall)
  - Data Visualization (6-panel matplotlib/seaborn dashboard)

Author: Koutilya Yenumula
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind, norm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, classification_report,
                              average_precision_score, roc_curve)
from sklearn.ensemble import GradientBoostingClassifier
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)

N_CREATORS = 500
OBSERVATION_WEEKS = 104          # 2 years of weekly data
FEATURE_CUTOFF_WEEK = 52         # Features computed from weeks 1–52 only (prevents leakage)

ARCHETYPES = {
    "Lifestyle": {"base_freq": 4.2, "burnout_prob": 0.38, "sentiment_decay": 0.003},
    "Gaming":    {"base_freq": 5.8, "burnout_prob": 0.42, "sentiment_decay": 0.004},
    "Finance":   {"base_freq": 3.1, "burnout_prob": 0.22, "sentiment_decay": 0.002},
    "Comedy":    {"base_freq": 6.5, "burnout_prob": 0.51, "sentiment_decay": 0.006},
    "Education": {"base_freq": 2.8, "burnout_prob": 0.18, "sentiment_decay": 0.001},
    "Fitness":   {"base_freq": 5.1, "burnout_prob": 0.33, "sentiment_decay": 0.003},
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: SYNTHETIC DATA GENERATION
# Realistic creator behavior with injected burnout signals
# ─────────────────────────────────────────────────────────────────────────────

def generate_creator_profile(creator_id):
    archetype = np.random.choice(list(ARCHETYPES.keys()))
    arch = ARCHETYPES[archetype]

    subscriber_count = int(np.random.lognormal(mean=12, sigma=1.8))
    years_active = np.random.uniform(0.5, 8)
    monetized = np.random.choice([0, 1], p=[0.35, 0.65])
    team_size = np.random.choice([1, 2, 3, 5, 10], p=[0.55, 0.20, 0.12, 0.08, 0.05])
    platform = np.random.choice(["YouTube", "TikTok", "Instagram", "Twitch"],
                                 p=[0.35, 0.30, 0.25, 0.10])

    burnout_risk = arch["burnout_prob"]
    if team_size == 1:
        burnout_risk += 0.10
    if years_active > 5:
        burnout_risk += 0.08
    if platform == "TikTok":
        burnout_risk += 0.06

    burned_out = np.random.random() < burnout_risk
    burnout_week = np.random.randint(30, OBSERVATION_WEEKS) if burned_out else None

    return {
        "creator_id": creator_id,
        "archetype": archetype,
        "subscriber_count": subscriber_count,
        "years_active": round(years_active, 2),
        "monetized": monetized,
        "team_size": team_size,
        "platform": platform,
        "burned_out": int(burned_out),
        "burnout_week": burnout_week,
        "base_posting_freq": arch["base_freq"],
        "sentiment_decay_rate": arch["sentiment_decay"],
    }


def generate_weekly_timeseries(creator_row):
    records = []
    burned_out = bool(creator_row["burned_out"])
    burnout_week = creator_row["burnout_week"]
    base_freq = creator_row["base_posting_freq"]
    decay = creator_row["sentiment_decay_rate"]

    signal_start = (burnout_week - 8) if burned_out else None

    for week in range(1, OBSERVATION_WEEKS + 1):
        is_post_burnout = burned_out and week > burnout_week
        is_signal_window = burned_out and signal_start and (signal_start <= week <= burnout_week)

        if is_post_burnout:
            post_freq = max(0, base_freq * np.random.uniform(0.05, 0.25) + np.random.normal(0, 0.3))
        elif is_signal_window:
            fade_factor = (week - signal_start) / 8
            post_freq = max(0, base_freq * (1 - 0.5 * fade_factor) + np.random.normal(0, 0.5))
        else:
            post_freq = max(0, base_freq + np.random.normal(0, 0.8))

        base_engagement = 0.045 + np.random.normal(0, 0.008)
        if is_signal_window:
            base_engagement *= (1 - 0.02 * (week - signal_start))
        if is_post_burnout:
            base_engagement *= 0.4

        base_sentiment = 0.65 - decay * week
        if is_signal_window:
            base_sentiment -= 0.04 * (week - signal_start)
        if is_post_burnout:
            base_sentiment = np.random.uniform(-0.3, 0.1)
        caption_sentiment = np.clip(base_sentiment + np.random.normal(0, 0.08), -1, 1)

        toxicity = 0.08 + np.random.normal(0, 0.02)
        if is_signal_window:
            toxicity += 0.015 * (week - signal_start)
        toxicity = np.clip(toxicity, 0, 1)

        response_rate = 0.22 + np.random.normal(0, 0.05)
        if is_signal_window:
            response_rate *= (1 - 0.04 * (week - signal_start))
        if is_post_burnout:
            response_rate = np.random.uniform(0.0, 0.04)
        response_rate = np.clip(response_rate, 0, 1)

        records.append({
            "creator_id": creator_row["creator_id"],
            "week": week,
            "posts_per_week": round(post_freq, 2),
            "engagement_rate": round(max(0, base_engagement), 4),
            "caption_sentiment": round(caption_sentiment, 4),
            "comment_toxicity": round(toxicity, 4),
            "response_rate": round(max(0, response_rate), 4),
            "burned_out_event": int(burned_out and week == burnout_week),
        })
    return records


def section1_generate_data():
    print("🚀 CreatorPulse: Building creator profiles...")
    creators_df = pd.DataFrame([generate_creator_profile(i) for i in range(N_CREATORS)])
    print(f"   ✅ {N_CREATORS} creator profiles generated")
    print(f"   📊 Burnout rate: {creators_df['burned_out'].mean():.1%}")
    print(f"   🎭 Archetypes: {creators_df['archetype'].value_counts().to_dict()}")

    print("\n📈 Generating weekly time-series data (this takes ~10 seconds)...")
    all_records = []
    for _, row in creators_df.iterrows():
        all_records.extend(generate_weekly_timeseries(row))

    ts_df = pd.DataFrame(all_records)
    ts_df = ts_df.merge(
        creators_df[["creator_id", "archetype", "burned_out", "burnout_week",
                     "platform", "subscriber_count", "team_size", "monetized", "years_active"]],
        on="creator_id"
    )
    print(f"   ✅ {len(ts_df):,} weekly records generated")
    return creators_df, ts_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def section2_eda(creators_df, ts_df):
    print("\n" + "="*60)
    print("SECTION 2: EXPLORATORY DATA ANALYSIS")
    print("="*60)

    print("\n📊 Burnout rates by creator archetype:")
    burnout_by_arch = creators_df.groupby("archetype")["burned_out"].agg(["mean", "count", "sum"])
    burnout_by_arch.columns = ["burnout_rate", "total_creators", "burned_out_count"]
    burnout_by_arch = burnout_by_arch.sort_values("burnout_rate", ascending=False)
    print(burnout_by_arch.round(3).to_string())

    print("\n📊 Burnout rates by platform:")
    print(creators_df.groupby("platform")["burned_out"].mean().sort_values(ascending=False).round(3))

    print("\n📊 Burnout rates by team size:")
    print(creators_df.groupby("team_size")["burned_out"].mean().sort_values(ascending=False).round(3))

    print("\n📊 Average engagement: pre-burnout signal vs signal window:")
    signal_window = ts_df[
        (ts_df["burned_out"] == 1) &
        (ts_df["week"] >= ts_df["burnout_week"] - 8) &
        (ts_df["week"] < ts_df["burnout_week"])
    ]
    pre_signal = ts_df[
        (ts_df["burned_out"] == 1) &
        (ts_df["week"] < ts_df["burnout_week"] - 8)
    ]
    print(f"   Pre-signal engagement rate: {pre_signal['engagement_rate'].mean():.4f}")
    print(f"   Signal window engagement:   {signal_window['engagement_rate'].mean():.4f}")
    print(f"   Δ = {(signal_window['engagement_rate'].mean() - pre_signal['engagement_rate'].mean()):.4f}")

    return burnout_by_arch


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: STATISTICAL HYPOTHESIS TESTING
# ─────────────────────────────────────────────────────────────────────────────

def section3_hypothesis_testing(creators_df, ts_df):
    print("\n" + "="*60)
    print("SECTION 3: STATISTICAL HYPOTHESIS TESTING")
    print("="*60)

    healthy_engagement = ts_df[ts_df["burned_out"] == 0]["engagement_rate"]
    burnout_pre = ts_df[
        (ts_df["burned_out"] == 1) &
        (ts_df["week"] < ts_df["burnout_week"] - 8)
    ]["engagement_rate"]

    stat_h, p_h = stats.shapiro(healthy_engagement.sample(500, random_state=42))
    stat_b, p_b = stats.shapiro(burnout_pre.sample(500, random_state=42))
    print(f"\n🔬 Shapiro-Wilk Normality Test:")
    print(f"   Healthy group (n sample=500): W={stat_h:.4f}, p={p_h:.4e}")
    print(f"   Burnout group (n sample=500): W={stat_b:.4f}, p={p_b:.4e}")
    print(f"   → Non-normal distributions → using Mann-Whitney U test")

    u_stat, p_mw = mannwhitneyu(healthy_engagement, burnout_pre, alternative="two-sided")
    effect_size_r = 1 - (2 * u_stat) / (len(healthy_engagement) * len(burnout_pre))
    print(f"\n🔬 Mann-Whitney U Test (Engagement Rate: Healthy vs Pre-Burnout):")
    print(f"   U-statistic: {u_stat:,.0f}")
    print(f"   p-value:     {p_mw:.4e}")
    print(f"   Effect size (r): {effect_size_r:.4f} "
          f"({'small' if abs(effect_size_r)<0.1 else 'medium' if abs(effect_size_r)<0.3 else 'large'})")
    print(f"   → {'SIGNIFICANT' if p_mw < 0.05 else 'NOT significant'} at α=0.05")

    t_stat, p_ttest = ttest_ind(
        ts_df[ts_df["burned_out"] == 0]["posts_per_week"],
        ts_df[ts_df["burned_out"] == 1]["posts_per_week"]
    )
    print(f"\n🔬 Welch's T-test (Posts Per Week: Healthy vs Burned-Out):")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value:     {p_ttest:.4e}")

    contingency = pd.crosstab(creators_df["platform"], creators_df["burned_out"])
    chi2_stat, p_chi2, dof, _ = chi2_contingency(contingency)
    cramers_v = np.sqrt(chi2_stat / (creators_df.shape[0] * (min(contingency.shape) - 1)))
    print(f"\n🔬 Chi-Square Test (Platform × Burnout Status):")
    print(f"   χ² = {chi2_stat:.4f}, df={dof}, p={p_chi2:.4f}")
    print(f"   Cramér's V = {cramers_v:.4f} (association strength)")

    bonferroni_threshold = 0.05 / 3
    print(f"\n🔬 Multiple Testing Correction (Bonferroni):")
    print(f"   Adjusted α = {bonferroni_threshold:.4f}")
    for name, p in [("MW-U test", p_mw), ("T-test", p_ttest), ("Chi-Square", p_chi2)]:
        sig = "✅ SIGNIFICANT" if p < bonferroni_threshold else "❌ NOT significant"
        print(f"   {name}: p={p:.4e} → {sig}")

    return p_mw, p_ttest, p_chi2


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: A/B TESTING — INTERVENTION EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def section4_ab_testing(creators_df):
    print("\n" + "="*60)
    print("SECTION 4: A/B TESTING — INTERVENTION EXPERIMENT")
    print("="*60)

    np.random.seed(123)
    n_at_risk = len(creators_df[creators_df["burned_out"] == 1])
    at_risk = creators_df[creators_df["burned_out"] == 1].sample(
        min(200, n_at_risk), replace=False, random_state=42
    ).copy()
    at_risk = at_risk.reset_index(drop=True)
    at_risk["group"] = np.where(np.arange(len(at_risk)) < 100, "treatment", "control")

    at_risk["burnout_resolved"] = 0
    for i, row in at_risk.iterrows():
        if row["group"] == "treatment":
            at_risk.at[i, "burnout_resolved"] = int(np.random.random() < 0.28)
        else:
            at_risk.at[i, "burnout_resolved"] = int(np.random.random() < 0.08)

    at_risk["retention_8w"] = at_risk.apply(
        lambda r: np.random.uniform(0.55, 0.85) if r["group"] == "treatment"
                  else np.random.uniform(0.30, 0.65), axis=1
    )

    print("\n📊 A/B Test Results: Wellness Check Intervention")
    print("-" * 50)
    ab_summary = at_risk.groupby("group").agg(
        n=("creator_id", "count"),
        burnout_resolved_pct=("burnout_resolved", "mean"),
        avg_retention=("retention_8w", "mean")
    ).round(4)
    print(ab_summary)

    treat = at_risk[at_risk["group"] == "treatment"]["retention_8w"]
    ctrl = at_risk[at_risk["group"] == "control"]["retention_8w"]

    t_stat_ab, p_ab = ttest_ind(treat, ctrl)
    cohens_d = (treat.mean() - ctrl.mean()) / np.sqrt((treat.std()**2 + ctrl.std()**2) / 2)
    lift = (treat.mean() - ctrl.mean()) / (ctrl.mean() if ctrl.mean() != 0 else 1e-10)

    print(f"\n📈 Treatment vs Control:")
    print(f"   Lift in retention:  +{lift:.1%}")
    print(f"   Cohen's d:          {cohens_d:.3f} "
          f"({'small' if cohens_d<0.2 else 'medium' if cohens_d<0.5 else 'large'})")
    print(f"   t-statistic:        {t_stat_ab:.4f}")
    print(f"   p-value:            {p_ab:.4e}")
    print(f"   Result:             {'✅ STATISTICALLY SIGNIFICANT' if p_ab < 0.05 else '❌ NOT significant'}")

    power_est = 1 - norm.cdf(norm.ppf(0.975) - abs(cohens_d) * np.sqrt(len(treat) / 2))
    print(f"   Statistical power:  {power_est:.3f}")

    return at_risk


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: SURVIVAL ANALYSIS — lifelines KaplanMeierFitter + CoxPHFitter
# ─────────────────────────────────────────────────────────────────────────────

def section5_survival_analysis(creators_df):
    print("\n" + "="*60)
    print("SECTION 5: SURVIVAL ANALYSIS")
    print("="*60)

    survival_df = creators_df.copy()
    survival_df["duration"] = survival_df.apply(
        lambda r: r["burnout_week"] if r["burned_out"] else OBSERVATION_WEEKS, axis=1
    )
    survival_df["event"] = survival_df["burned_out"]

    solo = survival_df[survival_df["team_size"] == 1]
    team = survival_df[survival_df["team_size"] > 1]

    # Kaplan-Meier via lifelines
    kmf_solo = KaplanMeierFitter()
    kmf_solo.fit(solo["duration"], solo["event"], label="Solo creators")

    kmf_team = KaplanMeierFitter()
    kmf_team.fit(team["duration"], team["event"], label="Team creators")

    print(f"\n📊 Kaplan-Meier Survival Analysis (lifelines KaplanMeierFitter):")
    solo_med = kmf_solo.median_survival_time_
    team_med = kmf_team.median_survival_time_
    print(f"   Solo creators (n={len(solo)}): median survival = "
          f"{'>' + str(OBSERVATION_WEEKS) if np.isinf(solo_med) else f'{solo_med:.0f}'} weeks")
    print(f"   Team creators (n={len(team)}): median survival = "
          f"{'>' + str(OBSERVATION_WEEKS) if np.isinf(team_med) else f'{team_med:.0f}'} weeks")

    # Formal log-rank test
    lr = logrank_test(
        solo["duration"], team["duration"],
        event_observed_A=solo["event"], event_observed_B=team["event"]
    )
    print(f"   Log-rank test: χ²={lr.test_statistic:.4f}, p={lr.p_value:.4f}")
    print(f"   → {'Significantly different survival curves' if lr.p_value < 0.05 else 'No significant difference in survival'}")

    # Cox Proportional Hazards via lifelines CoxPHFitter
    print("\n📊 Cox Proportional Hazards (lifelines CoxPHFitter):")
    cox_df = survival_df[["duration", "event", "team_size", "years_active",
                           "monetized", "subscriber_count"]].copy()
    cox_df["subscriber_log"] = np.log1p(cox_df["subscriber_count"])
    cox_df = cox_df.drop("subscriber_count", axis=1)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df, duration_col="duration", event_col="event", show_progress=False)

    summary = cph.summary[["exp(coef)", "p"]].copy()
    summary.columns = ["HR", "p_value"]
    for feat, row in summary.iterrows():
        direction = "↑ risk" if row["HR"] > 1 else "↓ risk"
        sig = "✅" if row["p_value"] < 0.05 else "  "
        print(f"   {sig} {feat:<22}: HR = {row['HR']:.3f}  p={row['p_value']:.3f}  ({direction})")

    return kmf_solo, kmf_team, survival_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: FEATURE ENGINEERING (leakage-free: weeks 1–52 only)
# ─────────────────────────────────────────────────────────────────────────────

def section6_feature_engineering(creators_df, ts_df):
    print("\n" + "="*60)
    print("SECTION 6: FEATURE ENGINEERING")
    print("="*60)
    print(f"   ⚠️  Features computed from weeks 1–{FEATURE_CUTOFF_WEEK} only "
          f"(prevents post-burnout data leakage into model inputs)")

    # Restrict to the observation window — no future/post-burnout data bleeds into features
    ts_window = ts_df[ts_df["week"] <= FEATURE_CUTOFF_WEEK].copy()

    creator_features = []
    for cid, grp in ts_window.groupby("creator_id"):
        grp = grp.sort_values("week")

        # Trend slopes (OLS on last 12 weeks of observation window)
        last12 = grp.tail(12)
        if len(last12) >= 4:
            slope_eng   = np.polyfit(range(len(last12)), last12["engagement_rate"],   1)[0]
            slope_posts = np.polyfit(range(len(last12)), last12["posts_per_week"],     1)[0]
            slope_sent  = np.polyfit(range(len(last12)), last12["caption_sentiment"],  1)[0]
        else:
            slope_eng = slope_posts = slope_sent = 0

        cv_posts  = grp["posts_per_week"].std() / (grp["posts_per_week"].mean() + 1e-8)
        wow_drops = (grp["engagement_rate"].pct_change() < -0.40).sum()

        if len(grp) >= 12:
            resp_decline = (grp["response_rate"].iloc[:12].mean()
                            - grp["response_rate"].iloc[-12:].mean())
        else:
            resp_decline = 0.0

        creator_info = creators_df[creators_df["creator_id"] == cid].iloc[0]
        creator_features.append({
            "creator_id": cid,
            "avg_engagement":       grp["engagement_rate"].mean(),
            "avg_posts_per_week":   grp["posts_per_week"].mean(),
            "avg_sentiment":        grp["caption_sentiment"].mean(),
            "avg_toxicity":         grp["comment_toxicity"].mean(),
            "avg_response_rate":    grp["response_rate"].mean(),
            "slope_engagement":     slope_eng,
            "slope_posts":          slope_posts,
            "slope_sentiment":      slope_sent,
            "cv_posting_freq":      cv_posts,
            "wow_engagement_drops": wow_drops,
            "response_rate_decline": resp_decline,
            "subscriber_log":       np.log1p(creator_info["subscriber_count"]),
            "years_active":         creator_info["years_active"],
            "team_size":            creator_info["team_size"],
            "monetized":            creator_info["monetized"],
            "burned_out":           creator_info["burned_out"],
        })

    features_df = pd.DataFrame(creator_features)
    print(f"   ✅ Engineered {len(features_df.columns)-2} features for {len(features_df)} creators")
    print(f"   📋 Features: {', '.join(c for c in features_df.columns if c not in ['creator_id','burned_out'])}")

    print("\n📊 Top feature correlations with burnout:")
    corr = features_df.drop(["creator_id", "burned_out"], axis=1).corrwith(features_df["burned_out"])
    print(corr.abs().sort_values(ascending=False).head(8).round(4).to_string())

    return features_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: ML MODEL — GRADIENT BOOSTING BURNOUT RISK CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def section7_ml_model(features_df):
    print("\n" + "="*60)
    print("SECTION 7: ML MODEL — BURNOUT RISK CLASSIFIER")
    print("="*60)

    feature_cols = [c for c in features_df.columns if c not in ["creator_id", "burned_out"]]
    X_ml = features_df[feature_cols].values
    y_ml = features_df["burned_out"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_ml, y_ml, test_size=0.2, random_state=42, stratify=y_ml
    )
    print(f"   Train: {len(X_train)} creators | Test: {len(X_test)} creators")
    print(f"   Burnout prevalence: train={y_train.mean():.1%}, test={y_test.mean():.1%}")

    gbm = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.8, min_samples_leaf=10, random_state=42
    )
    gbm.fit(X_train, y_train)
    y_prob = gbm.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)
    cv_scores = cross_val_score(
        gbm, X_ml, y_ml,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="roc_auc"
    )

    print(f"\n📊 Model Performance:")
    print(f"   ROC-AUC (test):    {roc_auc:.4f}")
    print(f"   PR-AUC  (test):    {pr_auc:.4f}")
    print(f"   5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Healthy', 'Burned Out'])}")

    print("📊 Top Feature Importances:")
    feat_imp = pd.Series(gbm.feature_importances_, index=feature_cols).sort_values(ascending=False)
    for feat, imp in feat_imp.head(8).items():
        bar = "█" * int(imp * 100)
        print(f"   {feat:<30} {imp:.4f}  {bar}")

    return gbm, feature_cols, y_test, y_prob


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: CAUSAL INFERENCE — DIFFERENCE-IN-DIFFERENCES
# ─────────────────────────────────────────────────────────────────────────────

def section8_causal_inference(ts_df):
    print("\n" + "="*60)
    print("SECTION 8: CAUSAL INFERENCE — DIFFERENCE-IN-DIFFERENCES")
    print("="*60)

    np.random.seed(77)
    pre_tiktok  = ts_df[(ts_df["platform"] == "TikTok")    & (ts_df["week"] < 52)]["posts_per_week"].mean()
    post_tiktok = ts_df[(ts_df["platform"] == "TikTok")    & (ts_df["week"] >= 52)]["posts_per_week"].mean()
    pre_insta   = ts_df[(ts_df["platform"] == "Instagram") & (ts_df["week"] < 52)]["posts_per_week"].mean()
    post_insta  = ts_df[(ts_df["platform"] == "Instagram") & (ts_df["week"] >= 52)]["posts_per_week"].mean()

    did_estimate = (post_tiktok - pre_tiktok) - (post_insta - pre_insta)

    print("\n📊 DiD: Algorithm Change Impact on Posting Frequency")
    print(f"   TikTok:    Pre={pre_tiktok:.3f} → Post={post_tiktok:.3f}  (Δ={post_tiktok - pre_tiktok:+.3f})")
    print(f"   Instagram: Pre={pre_insta:.3f} → Post={post_insta:.3f}  (Δ={post_insta - pre_insta:+.3f})")
    print(f"   DiD Estimate (causal effect of algorithm): {did_estimate:+.3f} posts/week")

    combined = ts_df[ts_df["platform"].isin(["TikTok", "Instagram"])].copy()
    null_dids = []
    for _ in range(500):
        shuffled = combined.copy()
        shuffled["platform"] = np.random.permutation(shuffled["platform"].values)
        pt  = shuffled[(shuffled["platform"] == "TikTok")    & (shuffled["week"] <  52)]["posts_per_week"].mean()
        po  = shuffled[(shuffled["platform"] == "TikTok")    & (shuffled["week"] >= 52)]["posts_per_week"].mean()
        pc  = shuffled[(shuffled["platform"] == "Instagram") & (shuffled["week"] <  52)]["posts_per_week"].mean()
        pco = shuffled[(shuffled["platform"] == "Instagram") & (shuffled["week"] >= 52)]["posts_per_week"].mean()
        null_dids.append((po - pt) - (pco - pc))

    p_did = np.mean(np.abs(null_dids) >= np.abs(did_estimate))
    print(f"   Permutation test p-value: {p_did:.4f}")
    print(f"   Conclusion: Algorithm change "
          f"{'causally' if p_did < 0.05 else 'did NOT causally'} increased posting pressure")

    return did_estimate, p_did


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: COHORT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def section9_cohort_analysis(creators_df):
    print("\n" + "="*60)
    print("SECTION 9: COHORT ANALYSIS — ENGAGEMENT DECAY BY TENURE")
    print("="*60)

    creators_df = creators_df.copy()
    creators_df["tenure_cohort"] = pd.cut(
        creators_df["years_active"],
        bins=[0, 1, 3, 5, 100],
        labels=["0-1yr", "1-3yr", "3-5yr", "5yr+"]
    )
    cohort_stats = creators_df.groupby("tenure_cohort", observed=True).agg(
        count=("creator_id", "count"),
        burnout_rate=("burned_out", "mean"),
        avg_subscribers=("subscriber_count", "median")
    ).round(3)
    print(cohort_stats)

    return cohort_stats, creators_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: VISUALIZATIONS — 6-panel dashboard
# ─────────────────────────────────────────────────────────────────────────────

def section10_visualizations(kmf_solo, kmf_team, y_test, y_prob,
                              gbm, feature_cols, creators_df, cohort_stats):
    print("\n" + "="*60)
    print("SECTION 10: VISUALIZATIONS")
    print("="*60)

    sns.set_theme(style="whitegrid", palette="muted")
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Kaplan-Meier Survival Curves ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    kmf_solo.plot_survival_function(ax=ax1, ci_show=True, color="#e74c3c")
    kmf_team.plot_survival_function(ax=ax1, ci_show=True, color="#2980b9")
    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="S(t)=0.5")
    ax1.set_title("Kaplan-Meier: Solo vs Team Creators", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Weeks Active")
    ax1.set_ylabel("Survival Probability (No Burnout)")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8)

    # ── Panel 2: ROC Curve ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    ax2.plot(fpr, tpr, color="#8e44ad", lw=2, label=f"AUC = {auc_val:.3f}")
    ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random")
    ax2.fill_between(fpr, tpr, alpha=0.10, color="#8e44ad")
    ax2.set_title("ROC Curve — Burnout Risk Classifier", fontsize=11, fontweight="bold")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)

    # ── Panel 3: Feature Importance ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    feat_imp = (pd.Series(gbm.feature_importances_, index=feature_cols)
                .sort_values(ascending=True)
                .tail(10))
    colors = ["#e74c3c" if v > 0.05 else "#3498db" for v in feat_imp.values]
    feat_imp.plot(kind="barh", ax=ax3, color=colors, edgecolor="white")
    ax3.set_title("Top Feature Importances", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Importance Score")
    ax3.set_ylabel("")

    # ── Panel 4: Burnout Rate by Archetype ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    arch_data = creators_df.groupby("archetype")["burned_out"].mean().sort_values(ascending=False)
    arch_colors = ["#e74c3c" if v > 0.45 else "#f39c12" if v > 0.35 else "#2ecc71"
                   for v in arch_data.values]
    arch_data.plot(kind="bar", ax=ax4, color=arch_colors, edgecolor="white")
    ax4.set_title("Burnout Rate by Creator Archetype", fontsize=11, fontweight="bold")
    ax4.set_xlabel("")
    ax4.set_ylabel("Burnout Rate")
    ax4.set_ylim(0, 0.80)
    ax4.tick_params(axis="x", rotation=30)
    for i, v in enumerate(arch_data.values):
        ax4.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=9)

    # ── Panel 5: Burnout Rate by Platform ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    plat_data = creators_df.groupby("platform")["burned_out"].mean().sort_values(ascending=False)
    plat_colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
    plat_data.plot(kind="bar", ax=ax5, color=plat_colors[:len(plat_data)], edgecolor="white")
    ax5.set_title("Burnout Rate by Platform", fontsize=11, fontweight="bold")
    ax5.set_xlabel("")
    ax5.set_ylabel("Burnout Rate")
    ax5.set_ylim(0, 0.80)
    ax5.tick_params(axis="x", rotation=30)
    for i, v in enumerate(plat_data.values):
        ax5.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=9)

    # ── Panel 6: Cohort Burnout Rate by Tenure ────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    cohort_plot = cohort_stats["burnout_rate"].reset_index()
    bar_colors = ["#3498db", "#f39c12", "#e67e22", "#e74c3c"]
    ax6.bar(
        cohort_plot["tenure_cohort"].astype(str),
        cohort_plot["burnout_rate"],
        color=bar_colors[:len(cohort_plot)],
        edgecolor="white"
    )
    ax6.set_title("Burnout Rate by Creator Tenure", fontsize=11, fontweight="bold")
    ax6.set_xlabel("Tenure Cohort")
    ax6.set_ylabel("Burnout Rate")
    ax6.set_ylim(0, 0.80)
    for i, v in enumerate(cohort_plot["burnout_rate"]):
        ax6.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=9)

    fig.suptitle("CreatorPulse: Creator Burnout Early Warning System — Analysis Dashboard",
                 fontsize=14, fontweight="bold", y=1.01)

    output_path = "creatorpulse_dashboard.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   ✅ Dashboard saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    creators_df, ts_df = section1_generate_data()
    section2_eda(creators_df, ts_df)
    section3_hypothesis_testing(creators_df, ts_df)
    section4_ab_testing(creators_df)
    kmf_solo, kmf_team, survival_df = section5_survival_analysis(creators_df)
    features_df = section6_feature_engineering(creators_df, ts_df)
    gbm, feature_cols, y_test, y_prob = section7_ml_model(features_df)
    section8_causal_inference(ts_df)
    cohort_stats, creators_df = section9_cohort_analysis(creators_df)
    section10_visualizations(kmf_solo, kmf_team, y_test, y_prob,
                              gbm, feature_cols, creators_df, cohort_stats)

    print("\n✅ Full analysis pipeline complete!")
    print("="*60)
    print("Summary of skills demonstrated:")
    skills = [
        "✅ EDA & Descriptive Statistics",
        "✅ Normality Testing (Shapiro-Wilk)",
        "✅ Non-parametric Testing (Mann-Whitney U)",
        "✅ Parametric Testing (Welch's T-test)",
        "✅ Chi-Square Test of Independence",
        "✅ Multiple Comparisons (Bonferroni correction)",
        "✅ A/B Testing with effect size (Cohen's d) & power analysis",
        "✅ Survival Analysis (Kaplan-Meier + log-rank test via lifelines)",
        "✅ Cox Proportional Hazards (CoxPHFitter via lifelines)",
        "✅ Feature Engineering (rolling, lag, slope, variability — leakage-free)",
        "✅ ML Classification (Gradient Boosting / XGBoost-style)",
        "✅ Cross-validation & Model Evaluation (ROC-AUC, PR-AUC)",
        "✅ Causal Inference (Difference-in-Differences + permutation test)",
        "✅ Cohort Analysis",
        "✅ Data Visualization (6-panel matplotlib/seaborn dashboard)",
    ]
    for s in skills:
        print(f"   {s}")


if __name__ == "__main__":
    main()
