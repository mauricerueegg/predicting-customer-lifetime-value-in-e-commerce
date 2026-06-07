# 06 – AI Usage Documentation

> This document summarises the AI usage practices applied in this project. The full, detailed version is in `notebook.ipynb` Section 7.

---

## 7.1 Role of AI

AI tools (Claude Code with Claude Sonnet) were used as a supporting tool throughout the project:

| Stage | AI Role |
|-------|---------|
| Project planning | Structured the notebook into sections, created implementation plan |
| Documentation | Generated initial drafts of all `/docs/` markdown files |
| Code generation | Implemented function-based code for models, evaluation, and SHAP |
| Feature engineering | Suggested new feature ideas (e.g., `total_spend_proxy`, `engagement_score`) |
| Debugging | Resolved SHAP compatibility issues (e.g., constructing `shap.Explanation` objects for waterfall plots) and fixed plotting issues |

**What AI did NOT do:**
- AI did not make final decisions about model choices — those were confirmed by the student
- AI did not validate that results are correct — all outputs were manually reviewed
- AI did not replace understanding — every code section was read and understood before use

---

## 7.2 Prompting Strategy

Three example prompts illustrate the strategy used throughout the project:

**Prompt 1 — Creating an instruction file for the AI:**
A detailed prompt specified all sections to include in a `CLAUDE.md` file: project description, dataset reference (all 11 columns with types and value ranges), domain context (e-commerce CLV challenges such as skewed distributions, customer heterogeneity, correlation vs. causation), model selection and assumptions, evaluation metrics, SHAP methodology, and code style requirements. This front-loaded all project context into a persistent reference file, making subsequent prompts shorter and more focused.

**Prompt 2 — Overfitting diagnosis and visualisation:**
Started with the concrete problem (Random Forest train RMSE ~59 vs. test RMSE ~143 — a gap of 140%) and requested specific diagnostics by name: train-vs-test comparison table, residual plots, VIF, Breusch-Pagan, Durbin-Watson, and Cook's distance. Explicitly instructed the AI not to suggest fixes yet — "understand before changing" — preventing the AI from jumping to solutions without characterising the issue first.

**Prompt 3 — Feature engineering through combination and decomposition:**
Structured the task into two strategies: (1) feature combination (multiply/divide existing features to capture interactions) and (2) feature decomposition (split or bin continuous features). Required business rationale and trade-off analysis per suggestion, with an explicit constraint about the small dataset size (1,100 rows). Despite this constraint, the AI still suggested more features than were ultimately useful — only two made it into the final model.

---

## 7.3 Orchestration & Timing

| Project Stage | AI Used? | Notes |
|---------------|----------|-------|
| Project planning & setup | Yes | Full plan created by AI, reviewed and approved |
| EDA (Section 2) | Yes | Visualisation code generated, plots reviewed manually |
| Feature engineering | Yes | Suggestions validated against data generating formula |
| Model training | Yes | Function implementations generated, hyperparameters reviewed |
| Evaluation | Partial | Results validated manually |
| SHAP analysis | Yes | SHAP code generated, results compared against known formula (manually) |
| Interpretation | Partial | Initial structure from AI, content written by student |

---

## 7.4 Validation of AI Outputs

| Output | Validation Method |
|--------|------------------|
| Dataset schema in docs | Cross-checked against actual DataFrame columns |
| Feature engineering logic | Verified `total_spend_proxy` correlates strongly with CLV in EDA |
| Model configurations | Compared to scikit-learn and XGBoost documentation |
| SHAP rankings | Compared against known CLV formula coefficients |
| Data leakage prevention | Verified scaler is fit on train only by checking mean/std |

---

## 7.5 Limitations of AI

Three key limitations were identified during the project:

**Cannot interpret results correctly — human validation is essential.**
The AI generated correct code for computing metrics and SHAP values, but its interpretation was shallow. It defaulted to recommending the model with the lowest RMSE without recognising that the simpler Linear Regression achieving nearly identical performance was the more meaningful finding — because it reveals the CLV relationship is fundamentally linear. SHAP rankings also needed manual cross-checking against the known data generating formula.

**AI always suggests adding more complexity — but more is not always better.**
The AI consistently proposed adding more derived features, interaction terms, and complex transformations. In practice, only two engineered features improved performance. The final results confirmed this: Linear Regression outperformed both tree-based models, demonstrating that complexity can hurt when the dominant signal is linear. The AI's bias toward complexity had to be actively counteracted.

**Overconfident on default hyperparameters.**
The AI initially suggested defaults (`n_estimators=100`, `max_depth=None`) without justification for a dataset of only 1,100 rows. When challenged, it acknowledged that `max_depth=None` allows fully grown trees — a known overfitting risk on small datasets — and recommended adjustments (reducing `max_depth`, adding `subsample`/`colsample_bytree`). AI tends to present defaults as universally good choices; the user must actively challenge these assumptions.

---

## 7.6 Reflection

Using AI as a coding assistant significantly accelerated project setup and boilerplate code generation. The AI was particularly effective with specific, well-scoped prompts — vague requests produced generic and sometimes incorrect code.

However, the project revealed that **AI cannot replace critical thinking about results**. The AI's interpretation was shallow: it focused on raw metrics without considering the broader picture (e.g., that a simple model matching a complex one is itself an important finding). The AI's **systematic bias toward complexity** had to be actively resisted — the final results validated the simpler approach. And **hyperparameter defaults** were presented without justification, requiring the student to explicitly challenge every assumption.

The most productive use of AI was not as an advisor, but as a fast code generator whose outputs need careful human review. In future projects, a more structured validation workflow — systematically challenging every interpretation, default, and complexity suggestion — would be the recommended approach.
