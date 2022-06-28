# stackoverflow-quality-rating
A machine learning enabled application to rate the question quality and suggest tags

# Project Overview

## Overview
#### Background
* **Customer:** Stackoverflow platform users
* **Goal:** To help the platform users in writing highquality questions with suggesting the right tags
* **Pains:**: Often poor question on stackoverflow creates a lot of problem for the user, many a times users gets suspended for repetitive behaviour for the same
* **Gains:**: Can help them in analyzing their question quality, and hence they can improve it. Other than that to improve their visibility, the suggested tags can be a great help too.

### Value Proposition
* **Product:** The Product will roll out in two phases.
              * In the first phase we will see the product will able to rate the question quality.
              * In the second phase we will roll out the feature where users will get suggested tags for their question to improve their visibility.
* **Alleviates:** It will help the developers in reducing their stackoverflow reputation by overcoming poor questions asked.
* **Advantages:** They can now easily check their quality and hence no worry of negative rating. as well as the correct set of tags helps.

### Objectives
* Create an automated pipeline to rate the question quality.
* Achieve > 85% for the classification problem on the defined metrics.
* Create another pipeline to suggest tags.

### Solutions
* **Core Features**
    * ML service to classify the question ratings.
    * Continual learning on the new data.
* **Secondary Features**
    * ML service to suggest the tags for each question.
* **Constraints**
    * Maintain low latency
    * Suggest only tags from the tags disctribution to avoide irrelevant tags
