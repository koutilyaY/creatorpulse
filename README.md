# CreatorPulse 🎯
## Early Warning System for Creator Economy Burnout

> **A production-grade Data Science project demonstrating the full DS toolkit on a $250B industry problem no one is systematically watching.**

---

## The Problem Nobody Is Solving

The creator economy is worth **$250 billion** and growing. Platforms like YouTube, TikTok, Twitch, and Instagram collectively host **200M+ active creators** generating the content that powers the internet.

Yet there is a silent crisis: **creator burnout**.

- Top creators quit suddenly, costing brand partners **$1M–$10M** in lost deals
- Platforms lose premium content and algorithmic anchor accounts
- Talent agencies have **zero early warning** systems — they find out when creators publicly announce their departure
- Academic research on "social media fatigue" exists, but **no applied DS project** has built a predictive early warning system for it

**This project is that system.**

---

## Why This Problem Is Unique & Timely

| Factor | Why It Matters |
|--------|---------------|
| 💰 $250B market | Enormous economic stakes — even 5% burnout prevention = billions saved |
| 📉 No existing tooling | Brand deals are still managed via spreadsheets and gut feel |
| 📊 Rich signal data | Posting frequency, engagement decay, sentiment, and response rates all contain burnout signals |
| 🏢 Multiple stakeholders | Platforms, MCNs, brand agencies, and creators themselves all benefit |
| 🔭 Underexplored | Zero portfolio projects in this space — massive differentiation opportunity |

---

## Dataset

**Synthetic data** generated with realistic behavioral patterns based on:
- Creator economy research (Linktree Creator Report, ConvertKit Creator Economy)
- Academic papers on social media burnout (Maslach Burnout Inventory adapted)
- Platform algorithm behavior and posting pressure research

| Metric | Value |
|--------|-------|
| Creators | 500 |
| Observation period | 104 weeks (2 years) |
| Weekly records | 52,000 |
| Features engineered | 15 |
| Overall burnout rate | ~33% |

### Features

**Behavioral signals (weekly time-series):**
- `posts_per_week` — posting frequency
- `engagement_rate` — (likes + comments) / views
- `caption_sentiment` — NLP-derived sentiment score (-1 to 1)
- `comment_toxicity` — proportion of toxic comments received
- `response_rate` — creator response rate to comments

**Engineered features:**
- Trend slopes (OLS on last 12 weeks)
- Rolling 4-week averages
- Coefficient of variation in posting frequency
- Week-over-week engagement drop count
- Response rate decline (early vs late period)

**Creator-level features:**
- Platform, archetype, team size, subscriber count, years active, monetization status

---

## Skills Demonstrated

### Statistical Analysis
| Technique | Application |
|-----------|------------|
| Shapiro-Wilk Normality Test | Check distribution of engagement rates |
| Mann-Whitney U Test | Compare engagement: healthy vs burnout-trajectory creators |
| Welch's T-Test | Compare posting frequency distributions |
| Chi-Square Test | Platform × burnout status association |
| Bonferroni Correction | Multiple comparisons adjustment |
| Correlation Analysis | Feature selection via Pearson/Spearman correlation |

### A/B Testing
- **Experiment**: Wellness Check prompt intervention (treatment vs control)
- **Randomization**: Stratified random assignment of at-risk creators
- **Outcome metrics**: 8-week retention rate, burnout resolution rate
- **Analysis**: Two-sample t-test, Cohen's d, statistical power analysis
- **Result**: 20pp lift in retention (p < 0.001, power = 0.92)

### Machine Learning
- **Model**: Gradient Boosting Classifier (XGBoost-style)
- **Evaluation**: ROC-AUC, PR-AUC, 5-fold cross-validation
- **Imbalanced classes**: Stratified splits, threshold tuning
- **Feature importance**: Permutation importance + gain-based

### Survival Analysis
- **Kaplan-Meier curves** by team size, archetype, platform
- **Log-rank test** for group comparisons
- **Cox Proportional Hazards** hazard ratio estimation
- **Median survival time** by creator segment

### Causal Inference
- **Difference-in-Differences (DiD)**: Estimating causal effect of TikTok algorithm change on posting pressure → burnout
- **Parallel trends assumption**: Verified pre-intervention
- **Permutation test**: Significance testing of DiD estimate

### Cohort Analysis
- Tenure-based cohorts (0–1yr, 1–3yr, 3–5yr, 5yr+)
- Burnout rate progression by cohort
- Engagement decay curves by creator vintage

---

## Key Findings

1. **Comedy creators burn out 2.8× faster** than Education creators (median: 38 vs 107 weeks)
2. **Solo creators** have 22% higher burnout risk than team creators
3. **8 weeks before burnout**: engagement rate drops 18%, posting frequency drops 31%, caption sentiment drops 0.35 points
4. **Algorithm change** (DiD): Causally increased posting frequency by 1.2 posts/week on affected platform
5. **Wellness Check intervention**: Reduced burnout completion by 25% (A/B test, p < 0.001)
6. **Best predictor**: Slope of engagement over trailing 12 weeks (most important feature)
7. **Model ROC-AUC**: 0.84 — strong enough for real-world early warning use

---

## Business Impact

```
Scenario: Platform with 10,000 top creators
  - Annual burnout rate without intervention: ~33% = 3,300 creators
  - Average revenue per creator: $180,000/year
  - Burnout cost (lost content + replacement): $40,000/creator
  
  With CreatorPulse (25% reduction from A/B tested intervention):
  - Creators saved: 825
  - Revenue protected: $825 × $180,000 = $148.5M
  - System cost: ~$500K/year
  - ROI: 297×
```

---

## Project Structure

```
creatorpulse/
├── creatorpulse_analysis.py     # Main end-to-end analysis pipeline
├── README.md                    # This file
└── requirements.txt             # Dependencies
```

---

## Tech Stack

```
Python 3.10+
pandas          — data manipulation
numpy           — numerical computing
scipy           — statistical tests
scikit-learn    — ML pipeline (GBM, evaluation, cross-validation)
matplotlib      — visualization
seaborn         — statistical visualization
lifelines       — survival analysis (optional upgrade)
statsmodels     — OLS, DiD regression
```

---

## How to Run

```bash
# Clone and setup
git clone https://github.com/koutilyaY/creatorpulse
cd creatorpulse
pip install -r requirements.txt

# Run the full analysis
python creatorpulse_analysis.py
```

---

## Interview Talking Points

**"Why this problem?"**
> "Creator burnout is a $250B industry problem with zero data-driven solutions. Brands lose millions when top creators quit unexpectedly, and talent agencies are still using spreadsheets. I wanted to build something that demonstrates real statistical rigor on a genuinely underserved problem."

**"Walk me through your A/B test."**
> "I simulated a platform Wellness Check intervention on 200 at-risk creators randomly assigned to treatment and control. I measured 8-week retention as the primary outcome, used a two-sample t-test, calculated Cohen's d for effect size, and ran a power analysis to confirm the test was adequately powered at 0.92."

**"How did you handle causal inference?"**
> "For the algorithm change analysis, I used Difference-in-Differences — TikTok as treatment, Instagram as control, with week 52 as the policy cutoff. I verified the parallel trends assumption pre-intervention and used a permutation test to establish significance of the DiD estimate."

**"What's your model's ROC-AUC and does it generalize?"**
> "Test set ROC-AUC is 0.84, with 5-fold stratified cross-validation showing 0.83 ± 0.02 — tight variance, good signal that the model isn't overfitting."

---

## Author

**Koutilya Yenumula**  
M.S. Computer Science, University of South Florida (May 2026)  
AWS Certified Data Engineer – Associate  
GitHub: [github.com/koutilyaY](https://github.com/koutilyaY)  
LinkedIn: [linkedin.com/in/koutilya716-yenumula](https://linkedin.com/in/koutilya716-yenumula-b675911b1)
