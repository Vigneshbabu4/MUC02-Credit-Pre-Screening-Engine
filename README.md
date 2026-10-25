# Credit Pre-Screening Engine: Loan Intake Optimisation

**Calibo AI Academy | Path 3 | Mini Use Case 02 (MUC02) | BFSI Retail Lending**
Prepared by: Vignesh

A machine-learning project that decides which personal-loan applications can skip the full manual credit review, either by being **auto-declined** (clearly bad) or **fast-tracked** (clearly good), while keeping the bank's risk limit.

---

## 1. Business problem

A mid-size private bank receives about **3,000 personal-loan applications a month**. Each one gets a **25-minute manual review**, which is **1,250 analyst hours a month**. Nearly half end in a decline, and many of those are obvious within a minute.

**Question:** Using six months of past decisions, how much of the manual queue can a model safely take over, and which model and thresholds should the bank adopt?

**Risk rule:** at most **5%** of the auto-decline lane may be applications the credit team approved, and at most **5%** of the fast-track lane may be applications the credit team declined.

## 2. Dataset

`MUC02_Credit_PreScreening_Dataset.csv`

- 12,000 applications (Apr to Sep 2026), 18 columns
- Target: `decision` (Approved / Declined). Overall approval rate is 52.9%
- Key terms: **FOIR** = (existing EMIs + proposed EMI) / monthly income; **NTC** = new-to-credit (no credit score)

## 3. Files in this project

| File | What it is |
|---|---|
| `Vignesh_MUC02_Path3_Notebook.ipynb` | Full notebook (first version, helper-based code) |
| `Vignesh_MUC02_Path3_Notebook_Simple.ipynb` | Same analysis, simple version: one command per cell |
| `Vignesh_MUC02_Path3_Notebook_Simple_v2.ipynb` | Simple version plus code and answers for the five recommendation points (**recommended**) |
| `MUC02_Notebook_Explanation.docx` | Word document explaining what was done in the notebook |
| `MUC02_Credit_PreScreening_Process.pptx` | 7-slide deck (orange, blue, black, white) explaining the process |
| `README.md` | This file |

## 4. What the notebook does

Every step has a plain-English explanation of the task, the code, and a findings cell.

**Part 1: Baseline pre-screener**

| Step | Content |
|---|---|
| 5.1 | Load, inspect and clean the data |
| 5.2 | Approval drivers: credit score and employment type |
| 5.3 | Engineer FOIR, loan-to-income and `is_ntc`; FOIR bands |
| 5.4 | Logistic regression (80/20 stratified split, `random_state=42`) |
| 5.5 | Auto-decline threshold against the 5% risk limit |
| 5.6 | Fairness check by gender |
| 5.7 | Interim briefing |

**Part 2: Model comparison and two-lane design**

| Step | Content |
|---|---|
| 5.8 | Rebuild the model as a single scikit-learn pipeline (raw CSV rows in, probability out) |
| 5.9 | 5-fold cross-validation of four models; one final test-set evaluation |
| 5.10 | Two-lane design (auto-decline / analyst review / fast-track) and monthly impact |
| 5.11 | Credit policy recommendation: five points, each with code and answer |
| 5.12 | Stretch: K-Means customer segments (k = 4) |

## 5. Data cleaning decisions

| Issue | Decision |
|---|---|
| 441 missing `monthly_income` (mostly Business Owners and Self-Employed) | Fill with the median income of the same employment type |
| 1,039 blank `credit_score` (all have zero credit history) | Keep; flag with `is_ntc`; fill with the median inside the model |
| 20 incomes above 350,000 (all multiples of 12, all Salaried) | Annual salaries typed as monthly; divide by 12 |

## 6. Key results

- **Credit-score cliff:** approval is about 3% below a score of 650 and 33% just above it.
- **FOIR:** approval drops to about 16% above 55%.
- **Logistic regression baseline:** 79.1% accuracy, but 17.7% of good applicants wrongly declined at the default 0.5 cut-off. Safe auto-decline threshold: 0.90 (15.2% of applications, 4.7% wrongly declined).
- **Best model:** Gradient Boosting, cross-validated ROC-AUC 0.889 against 0.868 for logistic regression.
- **Recommended cut-offs on P(approval):** auto-decline below **0.15**, fast-track at **0.88 or above**.
- **Lane shares (test set):** auto-decline 22.7%, analyst review 57.0%, fast-track 20.3%.
- **Monthly impact:** about 1,289 applications and 537 analyst hours freed (43%), against about 958 applications and 399 hours for logistic regression.

## 7. Risks and limits

- The `decision` column records what analysts decided, **not whether loans were repaid**, so the model copies past habits, including any bias.
- The 5% limits are reliable to only about **one point** (4.0% and 5.3% wrong on the test set).
- The model follows the historical gender gap (auto-decline: women 24.0%, men 22.1%) and gig workers are auto-declined at 40% against 20% for salaried staff, so outcomes must be monitored by gender and employment type.
- New-to-credit applicants and the high-income professionals segment should always stay with a human analyst.
- Recommended next step: a 4 to 8 week shadow-mode pilot before full rollout.

## 8. How to run

1. Put `MUC02_Credit_PreScreening_Dataset.csv` in the same folder as the notebook.
2. Install the requirements:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. Start Jupyter, open the notebook, and choose **Kernel > Restart & Run All**.

The notebook uses a fixed `random_state = 42`. The figures quoted in the markdown come from its own outputs, so if a library version changes the results, recheck them against the cell outputs.

## 9. Tech stack

Python 3, pandas, NumPy, scikit-learn (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, K-Means, Pipeline), matplotlib, seaborn, Jupyter.
