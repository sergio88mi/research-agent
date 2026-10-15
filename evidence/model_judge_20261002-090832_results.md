# Relevance screening checked by a second model: How effective is telemedicine for managing type 2 diabetes in adults?

- Brief: `output/brief-20261002-090832-how-effective-is-telemedicine-for-managi.json`
- Checked on 2026-10-03 by `gemini-3.6-flash`. The agent scored with `gemini-3.5-flash-lite`.
- Model requests made by this run of the check: 2 (answers already in the cache cost none).
- Papers: 65 scored by the agent, 36 of them kept (score 3 or more out of 5).
- What the second model saw for each paper: the research question, the sub-question that found it, the title, the year and the abstract as the agent's scoring prompt showed it (cut at 1,200 characters). 9 papers had no abstract. Ten papers to a request, per sub-question, in a shuffled order. It was not shown the agent's scores.
- Question put to it, as to a person judging by hand: "Would you keep this paper for this sub-question?" Yes: it addresses the sub-question, or is useful background for it. No: it only shares words with the sub-question, or is about something else. Unsure: the title and abstract do not allow a decision.

**These figures are agreement between two models.** They are not the precision of screening in the README's targets table, which is defined by people's judgements, and the two models, both Gemini, may share blind spots.

## Result

| | Second model: yes | Second model: no | Second model: unsure |
|---|---|---|---|
| Agent kept | 27 | 8 | 1 |
| Agent dropped | 3 | 25 | 1 |

- Kept papers the second model would also keep: 27 of the 35 it answered yes or no about (0.77).
- Dropped papers the second model would keep: 3.
- Agreement between the two, Cohen's kappa (unsure answers left out): 0.65.

## By sub-question

| Sub-question | Papers | Agent kept | Kept, second model yes | Share | Dropped, second model yes |
|---|---|---|---|---|---|
| 1 | 20 | 13 | 12 | 0.92 | 0 |
| 2 | 18 | 5 | 1 | 0.25 | 0 |
| 3 | 18 | 12 | 11 | 0.92 | 3 |
| 4 | 9 | 6 | 3 | 0.50 | 0 |

## Where the two models differ

### Kept by the agent, second model says no (8)

- Sub-question 1, agent's score 3: Telemedicine for the Management of Glycemic Control and Clinical Outcomes of Type 1 Diabetes Mellitus: A Systematic Review and Meta-Analysis of Randomized Controlled Studies (2017). Agent: "The abstract examines the impact of telemedicine on glycemic control, but focuses specifically on type 1 diabetes rather than type 2 diabetes." Second model: "This study evaluates telemedicine specifically in patients with type 1 diabetes rather than type 2 diabetes."
- Sub-question 2, agent's score 3: Diabetes self-management education for adults with type 2 diabetes mellitus: A systematic review of the effect on glycemic control (2015). Agent: "The abstract evaluates diabetes self-management education and its impact on glycemic control, but does not explicitly address remote monitoring or digital consultations." Second model: "This paper evaluates general diabetes self-management education without examining telemedicine, digital consultations, or remote monitoring tools."
- Sub-question 2, agent's score 3: Global Guideline for Type 2 Diabetes (2014). Agent: "There is no abstract available, but the title indicates global guidelines for type 2 diabetes management broadly." Second model: "The title indicates general practice guidelines for type 2 diabetes without mentioning telemedicine or remote patient monitoring."
- Sub-question 2, agent's score 3: Management of Hyperglycemia in Type 2 Diabetes, 2022. A Consensus Report by the American Diabetes Association (ADA) and the European Association for the Study of Diabetes (EASD) (2022). Agent: "The abstract discusses general management guidelines for type 2 diabetes but does not specifically focus on remote patient monitoring or digital consultations." Second model: "The consensus report addresses overall hyperglycemia management in type 2 diabetes without investigating telemedicine or remote patient monitoring."
- Sub-question 3, agent's score 3: A Randomized Trial Comparing Telemedicine Case Management with Usual Care in Older, Ethnically Diverse, Medically Underserved Patients with Diabetes Mellitus (2005). Agent: "This study evaluates telemedicine case management for patients with diabetes, primarily measuring clinical outcomes rather than patient or provider barriers and acceptance." Second model: "It evaluates clinical efficacy outcomes in a trial setting rather than assessing barriers, facilitators, or acceptance of telemedicine."
- Sub-question 4, agent's score 3: Determining if Telehealth Can Reduce Health System Costs: Scoping Review (2020). Agent: "The abstract investigates whether telehealth reduces health system costs compared to traditional care models, but discusses this broadly rather than specifically focusing on type 2 diabetes." Second model: "This paper focuses on broad health system costs of telehealth in general rather than specific cost-effectiveness for managing type 2 diabetes."
- Sub-question 4, agent's score 3: The cost-effectiveness of digital health interventions: A systematic review of the literature (2022). Agent: "The abstract reviews the cost-effectiveness of digital health interventions generally, serving as useful background rather than specific evidence for type 2 diabetes." Second model: "This systematic review addresses digital health interventions broadly without a specific focus on type 2 diabetes."
- Sub-question 4, agent's score 3: Effectiveness and Cost-Effectiveness of eHealth Interventions in Somatic Diseases: A Systematic Review of Systematic Reviews and Meta-Analyses (2014). Agent: "The abstract summarizes the effectiveness and cost-effectiveness of eHealth interventions across somatic diseases as a broad review, offering useful background context." Second model: "This review evaluates eHealth interventions across various somatic diseases generally rather than focusing specifically on type 2 diabetes."

### Dropped by the agent, second model says yes (3)

- Sub-question 3, agent's score 2: Analyzing older users’ home telehealth services acceptance behavior—applying an Extended UTAUT model (2016). Agent: "Since the abstract is missing, the paper is judged from the title alone as tangential because it addresses older users' home telehealth acceptance without mentioning diabetes." Second model: "The title explicitly focuses on analyzing user acceptance behavior regarding home telehealth services."
- Sub-question 3, agent's score 2: Personalized Telehealth in the Future: A Global Research Agenda (2016). Agent: "This paper outlines a global research agenda for personalized telehealth within chronic disease management and discusses implementation challenges, but is not specific to diabetes." Second model: "This paper discusses implementation challenges and adoption frameworks for personalized telehealth in chronic disease management."
- Sub-question 3, agent's score 2: Telemedicine in the OECD: An umbrella review of clinical and cost-effectiveness, patient experience and implementation (2020). Agent: "This umbrella review covers clinical effectiveness, patient experience, and implementation of telemedicine broadly across OECD countries without focusing on diabetes." Second model: "This umbrella review provides relevant findings on patient experience and implementation factors for telemedicine."

### Second model unsure (2)

- Sub-question 2, agent's score 3: Management of hyperglycaemia in type 2 diabetes, 2022. A consensus report by the American Diabetes Association (ADA) and the European Association for the Study of Diabetes (EASD) (2022). Agent: "There is no abstract available, but the title indicates it covers general management of hyperglycemia in type 2 diabetes rather than telemedicine specifically." Second model: "Without an abstract, it is unclear from the title alone whether this consensus report on hyperglycemia management addresses telemedicine, remote patient monitoring, or digital consultations."
- Sub-question 4, agent's score 1: Telemedicine in Complex Diabetes Management (2018). Agent: "The paper lacks an abstract, and the title alone only indicates complex diabetes management via telemedicine without mentioning cost-effectiveness." Second model: "The title indicates the paper is about telemedicine in diabetes management, but without an abstract, it is unclear if it addresses cost-effectiveness."

## What this shows and what it does not

- How often a second, larger model agrees with the agent's screening of the papers it found, judged from the same titles and abstracts. Not whether either model is right: that needs people.
- Not the search: whether the right papers were found at all (recall of retrieval) needs an independent list, such as the included studies of a published systematic review.
- One run on one research question.
