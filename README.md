# Bank Marketing Lead Scoring System

## Business Problem
Outbound telemarketing campaigns for bank products are expensive and constrained by sales agent bandwidth. In traditional campaigns, calling every lead uniformly yields an average conversion rate of only ~11.5%. Random or unranked dialing wastes agent time on low-intent prospects, inflates customer acquisition costs, and increases customer contact fatigue.

## Solution
A pre-call lead scoring system that predicts customer conversion propensity before an agent dials the phone. The model scores prospective leads and categorizes them into actionable priority tiers (High, Medium, Low), enabling outbound sales teams to focus their efforts on the prospects most likely to subscribe to a term deposit.

## Data
The model uses 4,521 customer records from the UCI Bank Marketing dataset. Features include customer demographics (age, job, education), financial indicators (account balance, credit default status, housing and personal loans), and previous campaign touchpoint history. To ensure a realistic, leakage-free operational system, call duration is strictly excluded because it is unknown prior to dialing.

## Business Value
- **Optimized Sales Operations:** Focuses finite call center capacity on top-converting prospects rather than working cold lists uniformly.
- **Lower Acquisition Costs:** Minimizes wasted dial time and outreach spend on low-probability contacts.
- **Actionable Lead Tiers:** Automatically assigns clear operational recommendations (e.g., immediate outreach with premium offers vs. deprioritizing outbound calls).
- **Improved Revenue Velocity:** Accelerates term deposit acquisitions by engaging high-propensity leads earlier in the campaign lifecycle.

## Machine Learning Approach
This is a binary classification and ranking problem with significant class imbalance (~88% non-conversions vs. ~12% conversions). A cost-sensitive **LightGBM** classifier was selected as the champion model over linear and ensemble baselines, tuned with an operational decision threshold ($\tau^* = 0.69$) to balance precision and recall.

## Result
On unseen test data, the model achieved a **3.25x conversion lift in the top 10% of dialed leads**, capturing **32.0% of all campaign conversions within the first 10% of calls** (and **48.7% within the top 20%**). For call center operations, this allows sales teams to capture nearly half of all deposit conversions while reducing outbound call volume by 80%.
