CLINICAL_TRIAL_PROMPTS = [
    "What are the demographics of patients enrolled in the clinical trial for [specific treatment]?",
    "How many patients were enrolled in the [specific trial name] and what was the criteria for enrollment?",
    "Can you summarize the primary and secondary outcomes of the clinical trial named [specific trial name]?",
    "What adverse events were reported during the [specific trial name]? Please summarize the findings.",
    "How does the efficacy of [treatment A] compare to [treatment B] based on recent trial results?",
    "What are the different phases of clinical trials and what type of data is collected at each phase?",
    "Can you identify any trends in patient dropout rates over the past five clinical trials?",
    "Based on the data, what recommendations can you provide for improving recruitment strategies for future trials?",
    "Perform a statistical analysis of the results from [specific trial] and provide insights on its significance.",
    "What insights can you provide regarding the overall success rates of clinical trials in the past year?"
]

PROMPT_QUALITY_OF_LIFE = (
    "Can you provide a comprehensive summary of the findings related to quality of life improvements "
    "for patients reported in the clinical trial named [specific trial name]? This should include "
    "details on any validated scales or questionnaires used to measure quality of life, specific areas "
    "where improvements were noted (such as physical health, mental well-being, social functioning), "
    "and any significant variations across different patient demographics or subgroups."
)

PROMPT_PLACEBO_EFFECT = (
    "How was the placebo effect managed and accounted for in the clinical trial investigating [specific treatment]? "
    "Please include information on the design measures used to mitigate placebo impact, such as blinding methods, "
    "control groups, and the proportion of patients who received a placebo. Additionally, summarize any observed "
    "placebo-related outcomes and their potential influence on the overall results."
)

PROMPT_ADVERSE_EVENT_DROPOUTS = (
    "What was the dropout rate in the clinical trial named [specific trial name] due to adverse events? "
    "Please provide a breakdown of the types and severity of adverse events reported, the proportion of patients "
    "who discontinued participation due to these events, and any notable differences in dropout rates across "
    "treatment and control groups. Also, mention if there were any patterns related to specific demographics or comorbidities."
)

PROMPT_DEMOGRAPHIC_IMPACT = (
    "How did patient demographic factors such as age, gender, ethnicity, and baseline health conditions "
    "influence the outcomes of the clinical trial named [specific trial name]? Please discuss any correlations or "
    "statistically significant findings that show variations in response rates, treatment efficacy, or side effects "
    "based on demographic attributes, and highlight any specific subgroups that showed distinct trends."
)

PROMPT_DATA_INTEGRITY = (
    "What procedures and methods were implemented in the clinical trial named [specific trial name] to ensure data "
    "integrity and reliability? This should include information on quality control measures, data collection protocols, "
    "blinding techniques, auditing processes, and any technology or software used to maintain accuracy in data reporting. "
    "Also, describe any challenges encountered and how they were addressed to preserve data quality."
)
