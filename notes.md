## Notes

### Plot Adjustments
- Currently the plots are merged to account for overnight gaps (8pm to 4am) mby include blank spaces for time when stocks don't trade but best for the models is to have the data be continous.
- No linear interpolation during these gaps.


###  Analysis of  the analysis: KEY FINDINGS

#### 1. Cumulative Model Performance Overview  1H (model_comparison_1h.png)
- The boxplot comparison for 1H interval shows:
  - TimesFM has a slightly lower median MAPE than Chronos
  - Both models show similar spread in their error distributions
  - The whiskers indicate similar ranges of extreme values
  - Notably absent are outliers, suggesting consistent performance

#### 2. Interval-wise Analysis (combined_distributions.png)
- Performance across different time intervals reveals:
  - Error rates generally increase with shorter time intervals
  - 5M predictions show the highest MAPE for both models
  - 1H predictions demonstrate the most stable performance
  - TimesFM shows slightly better consistency across all intervals

#### 3. Error Volatility Analysis (error_volatility_comparison_*.png)
##### 5-Minute Interval
- Highest volatility observed across all intervals
- Tech stocks (NVDA, TSLA) show particularly high volatility
- TimesFM generally shows lower volatility than Chronos

##### 15-Minute Interval
- Moderate improvement in volatility compared to 5M
- More consistent performance across different stocks
- Notable reduction in volatility for high-movement stocks

##### 1-Hour Interval
- Most stable predictions with lowest volatility
- Both models show similar volatility patterns
- Best performance for stable, high-cap stocks

#### 4. Directional Accuracy (directional_accuracy_comparison_*.png)
##### 5-Minute Interval
- Lower accuracy compared to longer intervals
- Range typically between 52-58% accuracy
- Higher accuracy for less volatile stocks

##### 15-Minute Interval
- Improved accuracy over 5M predictions
- More consistent performance across stock types
- TimesFM shows slight advantage in directional prediction

##### 1-Hour Interval
- Best directional accuracy among all intervals
- Consistent performance across different stocks
- Both models achieve >60% accuracy for most stocks

### Statistical Significance Analysis

#### MAPE Significance Tests
1. 5-Minute Interval
   - Significant difference between models (p < 0.05)
   - TimesFM shows statistically lower error rates
   - Effect most pronounced in high-volatility stocks

2. 15-Minute Interval
   - Moderate statistical significance (p < 0.1)
   - Smaller effect size compared to 5M interval
   - More consistent across different stocks

3. 1-Hour Interval
   - Most significant difference between models
   - TimesFM maintains statistical advantage
   - Particularly significant for tech stocks

#### Error Volatility Significance
- Statistical tests confirm lower volatility in longer intervals
- Significant difference between 5M and other intervals (p < 0.01)
- Model differences most significant in short-term predictions

#### Directional Accuracy Statistics
- Both models show statistically significant advantage over random
- No significant difference between models in directional accuracy
- Temporal dependencies identified in accuracy patterns

### Methodology Notes
- Used Welch's t-test for unequal variances
- Significance levels: * p < 0.05, ** p < 0.01, *** p < 0.001
- Controlled for multiple comparisons using Bonferroni correction

### Key Findings
1. Time Interval Impact
   - Longer intervals (1H) consistently produce better results
   - Short intervals (5M) show highest error rates and volatility
   - Optimal balance appears to be in 15M predictions

2. Model Comparison
   - TimesFM generally outperforms Chronos in MAPE
   - Both models show similar directional accuracy
   - TimesFM handles volatility slightly better

3. Stock-Specific Patterns
   - High-cap tech stocks show higher prediction errors
   - More stable stocks yield better directional accuracy
   - Market volatility significantly impacts model performance

### Conclusion
- Discuss why the model effectively learns trends but struggles with capturing volatility.
- Highlight potential reasons such as data limitations, model architecture, or feature engineering.
- Suggest improvements or alternative approaches to address volatility learning.

### Recommendations
1. Model Selection
   - Use Chronos for general predictions
   - Consider Chronos for specific high-volatility scenarios
   - Implement ensemble approach for critical predictions

2. Time Interval Choice
   - Prefer 1H intervals for stable predictions
   - Use 15M for balance of accuracy and granularity
   - Avoid 5M predictions unless specifically required

3. Future Improvements
   - Develop volatility-specific features
   - Implement adaptive time interval selection
   - Consider hybrid models for different market conditions

