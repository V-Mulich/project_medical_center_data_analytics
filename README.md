Проект: **Анализ изменения цены лечения в медицинской клинике**

dynamics_of_cost_and_exchange_rate.png

![][id]
⋮
[id]: dynamics_of_cost_and_exchange_rate.png "title"

**Описание проекта:**
Провел анализ изменения цен в медицинской клинике за 2022 год относительно 2021 года.

**Цель проекта:**
Определить ключевые факторы, влияющие на изменение цены лечения.

**Задачи:**
1. Сформировать гипотезы.
2. Провести исследовательский анализ данных.
3. Сделать выводы по гипотезам.
4. Написать общий вывод.

Используемые **библиотеки**:

* import pandas as pd
* import numpy as np
* import seaborn as sns
* import matplotlib.pyplot as plt
* import plotly.express as px

**Описание данных:**
* record_id - уникальный идентификатор строки данных;
* service_date - дата оказания медицинской услуги;
* service_name - наименование медицинской услуги;
* service_number - количество услуг;
* service_amount - сумма выплат (стоимость оказанных услуг в рублях);
* insured - уникальный идентификатор пациента;
* sex_id - пол пациента;
* age_for_service_date - возраст пациента.

Прежде чем начать анализ данных, предположил, что стоимость лечения возросла в 2022 году.

**Выдвинутые гипотезы:**
1. Стоимость лечения увеличилась из-за общего уровня инфляции в стране в 2022 году. (гипотеза 
2. Увеличение среднего возраста клиентов привело к росту стоимости лечения.
3. Влияние роста курса доллара на повышение стоимости лечения.

**Выводы по первой гипотезе - не отвергается:** \
Средний чек в 2022 году составил 1195 рублей, что на 4.9% больше, чем в 2021 году (1139 рублей). 
Средняя стоимость услуги выросла на 5.37%, что оказалось ниже уровня инфляции (11.94%) в России в 2022 году. 
Таким образом, стоимость услуг в 2022 году стала выше, но оставалась более доступной для клиентов.

**Выводы по второй гипотезе - отвергается:** \
График и расчет корреляции не выявили взаимосвязи между возрастом и стоимостью услуг. 
Следовательно, гипотеза о влиянии возраста клиентов на стоимость лечения отвергнута.

**Выводы по третьей гипотезе - отвергается.:** \
Корреляция между средней стоимостью услуг и курсом доллара была -0.438, указывая на обратную зависимость. 
Это указывает на то, что при увеличении курса доллара средняя стоимость услуг снижается. 
Вывод: повышение курса доллара коррелирует с уменьшением стоимости медицинских услуг.

**Общий вывод:** \
Рост стоимости услуг в 2022 году объясняется не только инфляцией, но и другими факторами. 
Влияние возраста клиентов на стоимость услуг не подтверждено, в то время как уменьшение 
стоимости услуг при росте курса доллара представляет собой интересную тенденцию. 
Однако стоит помнить, что корреляция не всегда свидетельствует о причинно-следственной 
связи, и другие факторы могут также влиять на стоимость лечения.
____________________________________________________________________________________________

**In English**

Project: **Analysis of Treatment Cost Changes in a Medical Clinic**

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
