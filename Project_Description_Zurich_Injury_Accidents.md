# Predicting Injury Accidents in the City of Zurich

Group Number 7,  Members: Jenny Braunstorfer, Anastasija Stirbanovic, Lydia Kristina Solomon, Kangyue Wang, Elisa Stepanski

Using police-registered road accidents in the city of Zurich, we predict whether an accident involves injuries or only property damage, based on the characteristics of the situation: place, time, road type and the road users involved. The result helps traffic-safety authorities rank situations by their risk of injury and decide where and when to focus prevention.

## 1. Problem statement

### Who would use the result, and for which decision?

The traffic-safety units of the city of Zurich, i.e. the city police and the traffic department (Dienstabteilung Verkehr). Their resources for prevention, such as traffic controls, awareness campaigns and patrols, are limited. They must decide where and when to focus prevention: at which locations, at which times, and for which groups of road users.

### What should improve, and how would we measure it?

Prevention resources should be concentrated on the situations in which accidents most often lead to injuries. We want to measure how well the model ranks accidents by their risk of injury: among the accidents the model rates as most risky (e.g. the top 10% or 20%), what share actually involved injuries, and what share of all injury accidents is captured there. We compare this with the baselines.

### What will the model predict, for which unit, at what moment, with what information?

- Target: the probability that an accident involves any injury (fatal, severe or light), versus property damage only.

- Unit: one police-registered road accident in the city of Zurich.

- Cutoff: the planning moment, e.g. at the end of a year, when prevention for the following year is planned. The model is trained on accidents up to that moment and evaluated on later accidents.

- Information: characteristics that describe the situation of an accident and that an authority can target with prevention: location, hour, weekday, month, road type, and whether a pedestrian, bicycle or motorcycle is involved. In addition, we derive three features: darkness (from month and hour), rush hour or weekend night, and distance to the city center.

- Accident type (e.g. rear-end collision, skidding) is closely related to the outcome. We therefore plan to compare models without and with accident type to see how much the predictions depend on it.

### What can the result not tell the decision-maker?

- The overall risk of an accident happening cannot be predicted by us. We can’t say whether an accident will happen or not. We try to predict probability of likeliness that a given accident involve es injuries.

- How severe the injuries are. We deliberately do not predict severity levels. Fatal accidents are too rare to predict reliably (64 in 2016–2025, 0.12%). Severe and fatal accidents together make up only 3.7%, below the 10% threshold, and would require techniques for imbalanced data which we want to avoid.

- Whether prevention works. The model ranks situations by risk; it does not tell whether controls or campaigns would actually reduce injuries. That is a causal question and out of scope.

- Accidents not reported to the police, or medical facts. Many minor accidents, especially bicycle falls might never be registered, so we are dealing with a somewhat selected sample.

- Factors like more Tempo 30 zones, more e-bikes and the unusual traffic in the COVID years (2020–2021) mean that past data may not fully describe the future.

## 2. Data

Source: City of Zurich open data, "Polizeilich registrierte Verkehrsunfälle auf dem Stadtgebiet Zürich seit 2011" (https://data.stadt-zuerich.ch/dataset/sid_dav_strassenverkehrsunfallorte).

Observations and time span: the file contains 70,767 accidents from 2011 to 2025. We use the years 2016–2025: 52,899 accidents, of which 12,716 involve injuries (24.0%). Type: cross-section; each row is a different accident, observed once.

Why we leave out 2011–2015: the number of recorded property-damage accidents rises sharply between 2014 and 2016 (from about 2,200–2,300 to about 4,300 per year), while the number of injury accidents stays stable. This points to a change in how accidents were recorded. Including those years would teach the model a share of injury accidents that no longer applies.

Target variable: AccidentSeverityCategory, recoded as binary: injury (fatal, severe or light injuries) vs. property damage only.

Minimum requirements:

- About 1,000 observations required: we have 52,899.

- At least about 50 cases of the rarer class: we have 12,716 injury accidents.

- The rarer class makes up 24.0%, above 10%, so that special techniques for dealing with imbalance could probably be avoided.

Features:

- From the data (10): accident type (compared without and with), pedestrian, bicycle and motorcycle involved, road type, east and north coordinates, month, weekday, hour. Year is in the data but not used as an input (see section 3).

- Derived by us (3): darkness (from month and hour, using sunrise and sunset times in Zurich), rush hour or weekend night, distance to the city centre.

- (The file has no exact date, only year, month, weekday and hour, so daily weather cannot be matched to individual accidents.)

Existing analyses: a UZH master's thesis and a related GitHub project use this dataset to model where bicycle accidents occur and how infrastructure such as road width or tram tracks relates to their severity, to support city planning. Our project is different: we consider all accidents rather than only bicycle accidents, we predict whether an accident involves any injury from the characteristics of the situation, and we use the result to rank situations for prevention. We make no claims about the effects of infrastructure or other causes.

## 3. First plan

Split: by time, because the model is used to plan for future years and conditions change over time: training 2016–2022 (37,958 accidents), validation 2023–2024 (10,227) to compare models and feature sets, and test 2025 (4,714), used only once at the end. Year is therefore not used as an input, since the model cannot learn anything about future years from it.

Baselines:

- Majority class: always predict "no injury". This reaches about 75% accuracy while missing every injury accident, which shows why accuracy is not a suitable metric here.

- Simple rule: predict "injury" if a pedestrian, bicycle or motorcycle is involved.

Model: logistic regression. We compare:

- the two baselines against logistic regression;

- logistic regression with different feature sets: without accident type, with accident type, and with the derived features;

- logistic regression with and without regularisation.

Metrics: precision and recall for injury accidents among the top-ranked accidents (e.g. top 10% and 20%)

Leakage and sanity checks:

- No input may be an outcome of the accident (hence the comparison without and with accident type).

- Scaling and encoding are fitted on the training data only.

- We check for missing values, coordinates outside the city, and whether the predicted probabilities are plausible (e.g. higher when a pedestrian or motorcycle is involved).

- We check whether the model performs differently in the COVID years and whether its performance is stable over time.
