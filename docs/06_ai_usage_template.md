# 06 – AI Usage Documentation Template

> Fill in this section in notebook.ipynb (Section 7). Replace all `[placeholder]` text with your actual content. Add more example prompts if you used more.

---

## 7. AI Usage Documentation

*Note: This section documents the use of AI tools (Claude Code / Claude Sonnet) in the development of this project. AI is treated as a supporting tool — all outputs were critically evaluated, verified, and adapted by the student.*

---

### 7.1 Role of AI

AI tools (specifically Claude Code with the Claude Sonnet model) were used for the following tasks in this project:

| Stage | AI Role |
|-------|---------|
| Project planning | Helped structure the notebook into sections, identify required steps, and create `/docs/` documentation |
| Feature engineering | Suggested engineered features (`total_spend_proxy`, `engagement_score`) and explained the rationale |
| Code implementation | Generated function-based implementations for model training, evaluation, and SHAP analysis |
| Documentation | Created initial drafts of `/docs/` markdown files explaining methodology choices |
| Debugging | [Describe any debugging assistance you received] |

**What AI did NOT do:**
- AI did not make final decisions about model choices — those were confirmed by the student
- AI did not validate that results are correct — all outputs were manually reviewed
- AI did not replace understanding — every code section was read and understood before use

---

### 7.2 Prompting Strategy

**Example Prompt 1 — Project planning:**
> "Help me create the plan and updated CLAUDE.md file. I want to predict customer lifetime value using regression and XAI. My focus is Focus E: Supervised Learning: Regression and XAI. Ask me if you have questions."

*Why this prompt worked:* It was specific about the focus area, gave the full context of the project, and invited follow-up questions. This allowed the AI to ask clarifying questions (e.g., which non-linear model, which language) rather than making assumptions.

**Example Prompt 2 — Documentation:**
> "Create the /docs/ markdown files for the project. Each file should document why and how something is done, not just what."

*Why this prompt worked:* The emphasis on "why and how" steered the AI away from shallow descriptions toward explanatory content with rationale — which is the most valuable part for a student project.

**Example Prompt 3 — [Add your own example]:**
> "[Your prompt here]"

*Why this prompt worked:* [Your explanation here]

---

### 7.3 Orchestration & Timing

| Project Stage | AI Used? | How |
|---------------|----------|-----|
| Initial setup (CLAUDE.md, docs) | Yes | Created plan, updated CLAUDE.md, created 7 docs files |
| EDA (Section 2) | [Yes/No] | [Describe] |
| Feature engineering (Section 4.1) | [Yes/No] | [Describe] |
| Model training (Section 4.2–4.3) | [Yes/No] | [Describe] |
| Evaluation (Section 4.4) | [Yes/No] | [Describe] |
| SHAP analysis (Section 4.5) | [Yes/No] | [Describe] |
| Results & reflection (Sections 5–6) | [Yes/No] | [Describe] |

---

### 7.4 Validation of AI Outputs

All AI-generated outputs were validated as follows:

| Output Type | Validation Method |
|-------------|------------------|
| Code | Executed and checked for errors; reviewed for correctness of logic |
| Documentation | Cross-checked against actual dataset schema and generation code |
| Feature engineering ideas | Verified against known CLV formula in the data generation code |
| SHAP explanations | Compared SHAP rankings to expected feature importance from the data generating process |
| Metric interpretations | Checked against scikit-learn documentation and course materials |

**Specific validation examples:**
- The `total_spend_proxy` feature was verified to be meaningful by checking correlation with CLV in Section 2
- SHAP feature rankings were compared to the known data generating formula to validate that the model learned correct patterns
- [Add more specific examples from your validation]

---

### 7.5 Limitations of AI

| Limitation | Example Encountered |
|------------|-------------------|
| Cannot run code | AI could not execute cells and verify that plots render correctly — this required manual testing |
| Stale knowledge | [Did any library API change cause issues? E.g., shap API differences] |
| Over-engineering | [Did AI suggest unnecessarily complex solutions that you simplified?] |
| Hallucinations | [Did AI make incorrect claims about library behaviour or data?] |
| No domain expertise | AI does not know the specifics of Swiss e-commerce or ZHAW grading criteria |

---

### 7.6 Reflection

> Write 3–5 sentences reflecting on what you learned about using AI in a data science workflow.

[Example: "Using Claude Code accelerated the initial project setup significantly — generating the seven documentation files would have taken hours manually. However, I found that AI is most useful as a starting point that requires refinement. The documentation templates needed to be personalised with my own observations from the data. The most important skill was knowing which AI outputs to trust (boilerplate structure, library syntax) versus which to verify carefully (feature engineering rationale, interpretation of SHAP results). I also noticed that giving the AI specific context (the full CLV formula, the exact column names) dramatically improved the quality of its output."]
