# ==========================================
# Tel Aviv University - R Final Project
# Itamar Ben Arie
# ==========================================

# * Dataset Description:
# This dataset consists of 416,809 tweets (AKA X posts), each labeled with an emotion category.
# The original labels are numeric (0-5), representing:
#   0 = Sadness
#   1 = Joy
#   2 = Love
#   3 = Anger
#   4 = Fear
#   5 = Surprise

# The goal of this study is to examine the relationship between emotions 
# and sentiment intensity in tweets. Specifically, we aim to:
# - Identify which emotion categories tend to have the most extreme sentiment scores.
# - Investigate whether sentiment extremity and sentiment polarity (AFINN score) 
#   influence tweet length.

# * Sentiment Score Calculation:
# The AFINN lexicon assigns a sentiment score to words on a scale from -5 (very negative) 
# to +5 (very positive). 
# A tweet’s sentiment score (Sentiment_AFINN) is calculated as the average sentiment 
# of the words it contains. 
# The absolute value of this score (Sentiment_Extreme) represents the degree of sentiment 
# intensity, regardless of whether the sentiment is positive or negative.

# This lexicon is widely used for sentiment analysis in text-based datasets.
# I will use AFINN to compute a sentiment score for each tweet by averaging the 
# sentiment values of the words in the tweet.

# * Research Question:
# Which emotions tend to have the most extreme sentiment scores?  
# Do sentiment extreme value and sentiment AFINN score predict tweet length?

# * Research Hypotheses:
# H1: Tweets with extreme sentiment scores (high Sentiment_Extreme) 
#     will tend to be shorter than tweets with moderate sentiment scores.
# H2: Tweets expressing negative emotions (Sadness, Anger, Fear)  
#     will have higher absolute sentiment scores (Sentiment_Extreme)
#     compared to those expressing positive emotions (Joy, Love, Surprise).
# H3: Tweets with higher Sentiment_AFINN scores will be more likely to be classified 
#     as expressing positive emotions (Label_Emotion_Type = "Positive").

# ==========================================
# Load required packages
# ==========================================
library(tidyverse)  # Data wrangling & visualization
library(ggplot2)    # Data visualization
library(dplyr)      # Data manipulation
library(tidytext)   # Text processing; New package
library(ggdist)     # Exploratory visualization
library(pROC)       # ROC Curve for logistic regression

# ==========================================
# Step A: Research Question & Data Exploration
# ==========================================

# Read the dataset
df <- read_csv("Emotions - Final Project.csv")

# Rename columns for clarity
colnames(df) <- c("id", "text", "label")

# Convert label to categorical variable with emotion names (6 categories)
df$Label <- factor(df$label, 
                   levels = c(0, 1, 2, 3, 4, 5), 
                   labels = c("Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"))

# Create a new variable - tweet length (word count)
df <- df |> mutate(text_length = str_count(text, "\\S+"))

# Display basic structure of dataset
glimpse(df)

# Plot the distribution of emotions
ggplot(df, aes(x = Label, fill = Label)) +
  geom_bar(alpha = 0.7) +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5) +  
  scale_y_continuous(labels = scales::comma) +  
  labs(title = "Distribution of Emotions in Tweets",
       x = "Emotion Category", y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")

# ==========================================
# Step B: Data Preprocessing
# ==========================================

# * Variable Definitions:
# - Multiple Linear Regression:
#   - Predictors: Sentiment_Extreme (absolute AFINN score) and Sentiment_AFINN
#   - Outcome: text_length (number of words in tweet)
# - Logistic Regression:
#   - Predictor: Sentiment_AFINN (AFINN score)
#   - Outcome: Label_Emotion_Type (Binary: Positive/Negative)

# ==========================================
# Compute sentiment score per tweet using AFINN

afinn_sentiment <- get_sentiments("afinn")

df_sentiment <- df |> 
  unnest_tokens(word, text) |>                 # From "tidytext" package
  inner_join(afinn_sentiment, by = "word") |> 
  group_by(id) |>  
  summarise(Sentiment_AFINN = mean(value))  

# Merge sentiment scores back into df
df <- df |> left_join(df_sentiment, by = "id")

# Replace NA sentiment scores with 0 (neutral)
df <- df |> mutate(Sentiment_AFINN = replace_na(Sentiment_AFINN, 0))

# Compute absolute sentiment extreme value (distance from zero)
df <- df |> mutate(Sentiment_Extreme = abs(Sentiment_AFINN))

# Create binary variable: Positive vs. Negative emotions for logistic regression
df <- df |> mutate(
  Label_Emotion_Type = ifelse(Label %in% c("Joy", "Love", "Surprise"), "Positive", "Negative"),
  Label_Emotion_Type = factor(Label_Emotion_Type, levels = c("Negative", "Positive"))  
)

# Sample a subset of the data including sentiment score
set.seed(42)
df_sample <- df |> sample_n(5000)

# ==========================================
# Classify tweets as "Extreme" or "Not Extreme"

# Define a function to classify tweets based on sentiment extremity
classify_extreme_tweet <- function(score, threshold = 4) {
  if (abs(score) >= threshold) {
    return("Extreme")
  } else {
    return("Not Extreme")
  }
}

# Apply the function to classify each tweet
df <- df |> mutate(Extreme_Category = sapply(Sentiment_AFINN, classify_extreme_tweet))

# * Visualization: Extreme vs. Not Extreme Tweets
df_extreme_pie <- df |> 
  count(Extreme_Category) |>  
  mutate(percentage = n / sum(n) * 100)  

ggplot(df_extreme_pie, aes(x = "", y = percentage, fill = Extreme_Category)) +
  geom_bar(stat = "identity", width = 1, color = "white") +  
  coord_polar("y", start = 0) +  
  geom_text(aes(label = paste0(round(percentage, 1), "%")), 
            position = position_stack(vjust = 0.5), color = "white", size = 5) +  
  labs(title = "Proportion of Extreme vs. Non-Extreme Tweets") + 
  theme_void() +  
  theme(legend.title = element_blank())  

# * Observation:
# - Only 1.2% of tweets received an AFINN absolute score of 4 or higher and are classified as extreme.

# ==========================================
# Sentiment Scores by Emotion

# Compute mean sentiment score per emotion (all six categories)
df_summary <- df |> 
  group_by(Label) |>  
  summarise(mean_sentiment = mean(Sentiment_AFINN, na.rm = TRUE),
            sd_sentiment = sd(Sentiment_AFINN, na.rm = TRUE),
            count = n()) |> 
  arrange(mean_sentiment)  

# Print the summary table
print(df_summary)

# * Visualization: Mean Sentiment Score by Emotion Type (6 Categories)
ggplot(df_summary, aes(x = reorder(Label, mean_sentiment), 
                       y = mean_sentiment, fill = Label)) +
  geom_col() +
  labs(title = "Mean Sentiment Score by Emotion",
       x = "Emotion",
       y = "Mean Sentiment Score (AFINN)") +
  theme_minimal()

# Identify the most extreme emotions (largest absolute sentiment scores)
df_extreme <- df_summary |> 
  mutate(abs_sentiment = abs(mean_sentiment)) |> 
  arrange(desc(abs_sentiment))  # Sorting by absolute values

# Print the extreme emotions summary
print(df_extreme)

# * Visualization: Mean Absolute Sentiment Score by Emotion
ggplot(df_extreme, aes(x = reorder(Label, abs_sentiment), 
                       y = abs_sentiment, fill = Label)) +
  geom_col() +
  labs(title = "Mean Absolute Sentiment Score by Emotion",
       x = "Emotion",
       y = "Mean Absolute Sentiment Score (|AFINN|)") +
  theme_minimal()

# ==========================================
# Comparing Positive vs. Negative Emotions

# Compute mean sentiment score per sentiment type (Positive/Negative)
df_extreme_type <- df |> 
  group_by(Label_Emotion_Type) |>  
  summarise(mean_sentiment = mean(Sentiment_AFINN, na.rm = TRUE),
            mean_abs_sentiment = mean(abs(Sentiment_AFINN), na.rm = TRUE), 
            sd_abs_sentiment = sd(abs(Sentiment_AFINN), na.rm = TRUE),  
            count = n()) |>  
  arrange(desc(mean_abs_sentiment))  # Sort by sentiment extremity

# Run an independent t-test to compare sentiment scores
t_test_results <- t.test(Sentiment_AFINN ~ Label_Emotion_Type, data = df)

# Print the results
print(t_test_results)

# * Visualization: Comparing Sentiment Score vs. Sentiment Extremity
ggplot(df_extreme_type, aes(x = Label_Emotion_Type, y = mean_abs_sentiment, fill = Label_Emotion_Type)) +
  geom_col(alpha = 0.7) +  # Use bars instead of points to emphasize contrast
  geom_errorbar(aes(ymin = mean_abs_sentiment - sd_abs_sentiment, 
                    ymax = mean_abs_sentiment + sd_abs_sentiment), 
                width = 0.2) +  # Error bars for variability
  labs(title = "Mean Absolute Sentiment Score (Positive vs. Negative)",
       x = "Sentiment Type",
       y = "Mean Absolute Sentiment Score (|AFINN|)",
       fill = "Sentiment Type") +
  theme_minimal()

# * Observations:
# - Joy, Love, and Surprise have the highest mean sentiment scores, 
#   while Anger, Sadness, and Fear have the lowest.
# - When looking at absolute sentiment extremity, positive emotions still tend to have higher values.  
# - Two Sample t-test: t = -523.68, p < 2.2e-16, confirming that positive emotions 
#   have significantly stronger sentiment scores than negative emotions.  
# * H2 NOT SUPPORTED: Contrary to our hypothesis, negative emotions do not have 
#   the highest sentiment extremity.  
# - Possible explanations:
#   (1) Lexicon bias — The AFINN lexicon may assign stronger sentiment values to positive words.  
#   (2) Expression intensity — People might express positive emotions more intensely in text.  
#   (3) Nuanced negativity — Negative emotions may be expressed in a more complex, varied way, 
#       leading to lower sentiment scores.  

# ==========================================
# Step C: Statistical Analysis
# ==========================================

# ==========================================
# 1. Multiple Linear Regression: 
#    Does extreme sentiment and sentiment AFINN score influence tweet length?

lm_text_length <- lm(text_length ~ Sentiment_Extreme * Sentiment_AFINN, data = df_sample)
summary(lm_text_length)

# * Observations:
# - Sentiment_Extreme has a significant negative effect on tweet length (β = -1.35, p < 0.001), 
#   meaning more extreme sentiment leads to shorter tweets.
# - Sentiment_AFINN has a significant positive effect (β = 1.71, p < 0.001), 
#   meaning more positive sentiment is linked to longer tweets.
# - The interaction term (β = -0.35, p < 0.01) suggests that as sentiment becomes more extreme, 
#   the positive effect of Sentiment_AFINN weakens.
# * H1 CONFIRMED: More extreme tweets are shorter, 
#   but the model explains only a small part of the variance (R² = 0.027).

# ==========================================
# 2. Binary Logistic Regression: 
#    Predicting positive/negative sentiment (as stated in 'label') using sentiment score (AFINN)

logistic_sentiment <- glm(Label_Emotion_Type ~ Sentiment_AFINN, data = df_sample, family = binomial)
summary(logistic_sentiment)

exp(coef(logistic_sentiment))

# * Observations:
# - Sentiment_AFINN has a strong positive effect on predicting positive emotions 
#   (β = 1.25, p < 0.001).
# - Odds ratio = 3.48, meaning each unit increase in Sentiment_AFINN 
#   triples the odds of a tweet being positive.
# - H3 CONFIRMED: Higher sentiment scores significantly increase the probability 
#   of a tweet being classified as positive.

# ==========================================
# 3. Plots for both Linear and Logistic Regressions

ggplot(df_sample, aes(x = Sentiment_Extreme, y = text_length, color = factor(Sentiment_AFINN > 0, labels = c("Negative", "Positive")))) +
  geom_point(alpha = 0.1) +
  geom_smooth(method = "lm", se = TRUE) +
  labs(title = "Effect of Sentiment Extreme on Tweet Length",
       x = "Sentiment Extreme (|AFINN Score|)",
       y = "Tweet Length (Words)",
       color = "Sentiment (AFINN)") +
  theme_minimal()

ggplot(df_sample, aes(x = Sentiment_AFINN, y = as.numeric(Label_Emotion_Type) - 1)) +
  geom_jitter(alpha = 0.1, width = 0.1) + 
  geom_smooth(method = "glm", method.args = list(family = "binomial"), color = "blue") +
  labs(title = "Logistic Regression: Sentiment AFINN Score vs. Positive Label Emotion",
       x = "Sentiment Score (AFINN)",
       y = "Probability of Positive Emotion") +
  theme_minimal()

# ==========================================
# 4. ROC Curve for Binary Logistic Regression

roc_curve <- roc(df_sample$Label_Emotion_Type, predict(logistic_sentiment, df_sample, type = "response"))

auc(roc_curve)

ggplot(data.frame(FPR = 1 - roc_curve$specificities, 
                  TPR = roc_curve$sensitivities), 
       aes(x = FPR, y = TPR)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = "ROC Curve for Logistic Regression Model",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal() +
  annotate("text", x = 0.5, y = 0.25, 
           label = paste("AUC =", round(auc(roc_curve), 3)), 
           color = "black", size = 5)

#  The ROC curve shows strong predictive power, with an AUC = 0.872.
#  This suggests that sentiment AFINN score is a highly effective predictor 
#  for distinguishing positive vs. negative tweets.

# ==========================================
# Final Summary of Findings
# ==========================================

# * Research Question 1: Which emotions tend to have the most extreme sentiment scores?
#   - Positive emotions (Joy, Love, Surprise) had higher sentiment scores 
#     than negative emotions (Anger, Sadness, Fear).
#   - Negative emotions did not show higher absolute sentiment values, contradicting H2.
#   - H2 NOT SUPPORTED.

# * Research Question 2: Do sentiment extreme value and sentiment AFINN score predict tweet length?
#   - Tweets with extreme sentiment values tend to be shorter.
#   - Sentiment_AFINN positively predicts tweet length, while Sentiment_Extreme negatively 
#     predicts it.
#   - H1 CONFIRMED.

# * Logistic Regression:
#   - Higher Sentiment_AFINN scores increase the probability of a tweet being classified as positive.
#   - Odds ratio = 3.48. Each unit increase in Sentiment_AFINN triples the odds 
#     of positive classification.
#   - H3 CONFIRMED.

# * Limitation:
#   - AFINN lexicon may have biases, overestimating positive sentiment intensity.

# ==========================================
# End of Script - Project Completed
# ==========================================