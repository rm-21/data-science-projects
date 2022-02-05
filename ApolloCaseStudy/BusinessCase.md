**Context**

Apollo Hospitals was established in 1983, renowned as the architect of modern healthcare in India. As the nation's first corporate hospital, Apollo Hospitals is acclaimed for pioneering the private healthcare revolution in the country.


As a data scientist working at Apollo 24/7, the ultimate goal is to tease out meaningful and actionable insights from Patient-level collected data.

You can help Apollo hospitals to be more efficient, to influence diagnostic and treatment processes, to map the spread of a pandemic.


One of the best examples of data scientists making a meaningful difference at a global level is in the response to the COVID-19 pandemic, where they have improved information collection, provided ongoing and accurate estimates of infection spread and health system demand, and assessed the effectiveness of government policies.

**How can you help here?**

The company wants to know:

• Which variables are significant in predicting the reason for hospitalization for different regions

• How well some variables like viral load, smoking, Severity Level describe the hospitalization charges

**Column Profiling**
**Age**: This is an integer indicating the age of the primary beneficiary (excluding those above 64 years, since they are generally covered by the government).
**Sex**: This is the policy holder's gender, either male or female
**Viral** **Load**: Viral load refers to the amount of virus in an infected person's blood
**Severity Level**: This is an integer indicating how severe the patient is
**Smoker**: This is yes or no depending on whether the insured regularly smokes tobacco.
**Region**: This is the beneficiary's place of residence in Delhi, divided into four geographic regions - northeast, southeast, southwest, or northwest
**Hospitalization charges**: Individual medical costs billed to health insurance

**Concept Used:**
* Graphical and Non-Graphical Analysis
* 2-sample t-test: testing for difference across populations
* ANOVA
* Chi-square

**How to begin**
* Import the dataset and do usual exploratory data analysis steps like checking the structure & characteristics of the dataset
* Try establishing a relation between the dependent and independent variable (Dependent “hospitalization charges” & Independent: Smoker, Severity Level etc)

* **Statistical Analysis:**
  * Prove (or disprove) that the hospitalization of people who do smoking is greater than those who don't? (T-test Right tailed)
  * Prove (or disprove) with statistical evidence that the viral load of females is different from that of males (T-test Two tailed)
  * Is the proportion of smoking significantly different across different regions? (Chi-square)
  * Is the mean viral load of women with 0 Severity level , 1 Severity level, and 2 Severity level the same? Explain your answer with statistical evidence (One way Anova)
* Set up Null Hypothesis (H0)
* State the alternate hypothesis (H1)
* Check assumptions of the test (Normality, Equal Variance). You can check it using Histogram, Q-Q plot or statistical methods like levene’s test, Shapiro-wilk test (optional)
* Please continue doing the analysis even If some assumptions fail (levene’s test or Shapiro-wilk test) but double check using visual analysis and report wherever necessary
* Set a significance level (alpha)
* Calculate test Statistics.
* Decision to accept or reject null hypothesis.
* Inference from the analysis
