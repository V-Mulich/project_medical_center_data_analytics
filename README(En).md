# Project: Analysis of Treatment Cost Changes in a Medical Clinic

**Project Description:**
Conducted an analysis of price changes in a medical clinic for the year 2022 compared to 2021.

**Project Objective:**
To identify key factors influencing the changes in treatment costs.

**Tasks:**
1. Formulate hypotheses.
2. Conduct exploratory data analysis.
3. Draw conclusions on the hypotheses.
4. Summarize findings.

**Libraries** used:
* import pandas as pd
* import numpy as np
* import seaborn as sns
* import matplotlib.pyplot as plt
* import plotly.express as px

**Data Description:**
* record_id - unique identifier for a data row;
* service_date - date of medical service;
* service_name - name of the medical service;
* service_number - quantity of services;
* service_amount - payment amount (cost of services provided in rubles);
* insured - unique patient identifier;
* sex_id - patient's gender;
* age_for_service_date - patient's age.

Before starting the data analysis, it was assumed that the cost of treatment increased in 2022.

**Hypotheses Formulated:**
1. The cost of treatment increased due to the overall inflation rate in the country in 2022.
2. The increase in the average age of clients led to an increase in the cost of treatment.
3. The impact of the dollar exchange rate growth on the increase in the cost of treatment.

**Conclusions on the First Hypothesis - not rejected:** \
The average bill in 2022 was 1195 rubles, which is 4.9% higher than in 2021 (1139 rubles).
The average cost of the service increased by 5.37%, which was lower than the inflation rate (11.94%) 
in Russia in 2022. Thus, the cost of services in 2022 became higher but remained more affordable for clients.

**Conclusions on the Second Hypothesis - rejected:** \
The graph and correlation calculation did not reveal a relationship between age and the cost of services.
Therefore, the hypothesis of the influence of clients' age on the cost of treatment is rejected.

**Conclusions on the Third Hypothesis - rejected:** \
The correlation between the average cost of services and the dollar exchange rate was -0.438, indicating an inverse relationship.
This suggests that as the dollar exchange rate increases, the average cost of services decreases.
Conclusion: the increase in the dollar exchange rate correlates with a decrease in the cost of medical services.

**Overall Conclusion:** \
The increase in the cost of services in 2022 is explained not only by inflation but also by other factors.
The influence of clients' age on the cost of services is not confirmed, while the reduction in the cost of 
services with the growth of the dollar exchange rate represents an interesting trend.
However, it is essential to remember that correlation does not always indicate a cause-and-effect 
relationship, and other factors may also influence treatment costs.

