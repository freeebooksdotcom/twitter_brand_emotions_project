# Tweet Sentiment Modeling
by Anton Haugen & Numen Rubino

### Sources 
-Database "Brand<br>
https://data.world/crowdflower/brands-and-product-emotions

## Introduction
Gauging the sentiment of social media buzz can help reorient a company's strategy for their product launches. What panels, opinions, and features get people the most excited about a product? And what causes negative reactions?

## Data

Provided by data world, the dataset "Brands and Product Emotions" features nearly 10,000 tweets scraped from twitter during 2011's SXSW festival. Noted for an increase in the presence of technology companies, that year's SXSW featured two major product launches by Apple and Google, an iPad launch at a popup Apple store and a recreation of the GooglePlex to help promote Android Gingerbread and other Google products. Alongside of these buzzworthy launches were several panels, talks, and parties where opinions about these brands circulated.

These 10,000 tweets were then classified by the curators to gauge emotional reaction of each tweet to the brand, according to whether the tweeter felt positive, negative, or neutral about the company mentioned. 

## EDA

Using these human-labels, we analyzed the class distribution of tweets, finding most fell into the neutral category, while the positive and negative reactions were second and third. We decided to start with a two classifier model as a result.
![Image](images/emotiondistribution.png?raw=true)

Both Apple and Google had comparative proportions of positive and negative tweets. However, Apple iPhone had a statistically significant higher proportion of negative reactions. Many of these negative reactioons had to do with iPhone 3G's short battery life and the lack of places to charge at that year's festival.
![Image](images/brandcomparison.png?raw=true)

To continue our EDA, we compared the frequency of terms from both positive and negative tweets. While iPad's popup store launch was praised in positive tweets, many of the negative tweets that criticized design were specifically about iPad's buttons and design. Other positive tweets to both brands praised temporary events like popups and parties in down town, while others complained about charging and long lines.  Interestingly, "rumors" more frequently appeared in positive tweets while "news" more frequently appeared in negative tweets.
![Image](images/word_freq_axes.png?raw=true)

To assist interpretability of the models we trained, we isolated words used exclusively in positive tweets and those used exclusively in negative tweets, sizing them according to their frequency in the entire corpus. 
![Image](images/pos_word_wordcloud.png?raw=true)
![Image](images/neg_word_wordcloud.png?raw=true)

Further analysis of terms that appeared in the positive word cloud exposed a slight error in our datacleaning and tokenization/lemmatization, 'case' was being used to promote free phonecases and was also being used in the misspelling of the word 'showcase' as "show case". The negative wordcloud highlighted a criticism of Apple's brand and design philosophy by one panelist as "fascist", a description that was quoted and retweeted multiple times across our dataset. "button" was a criticism tied to the hardware design of iPad2. 

To see what were the takeaways from SXSW 2011 for a product/marketing strategy, the successes and the flops, we also looked at the nouns that appeared most frequently in positive and negative tweets.
![Image](images/pos_noun_wordcloud.png?raw=true)
![Image](images/neg_noun_wordcloud.png?raw=true)
Many praised the art and music events around the festival, while in the negative nouns, many retweeted negative sentiments about what one panelist described as the "digital delegates." "Batterkiller" appeared as both a reference to iPhone3G's short battery life and to apps that drained battery life faster than others, such as the Twitter app. Based on our EDA, one possible marketing strategy for a future SXSW could be branded charging stations since many negative reactions were about batterylife.

## Feature Engineering/Preprocessing
spaCy was used for tokenization and lemmatization. We first classified negative and positive reactions as 0 and 1, and later when neutral reactions were included, we classified neutral, negative, and positive as 0, 1, and 2 respectively. Scikit learn's TF-IDF Vectorizer was used to quantify our terms within the corpus
To address class imbalance, upsampling was used as well as class-weight=balanced.
We first used accuracy as our evaluation metric for our models. Macro f1 was later employed to punish overpredicting the majority class. A dummy classifier scored a macro f1 of .3, so a good model would probably be better than .5.

## Modeling
For our two-class predictions, we used a Gaussian Naïve Bayes Classifier, Multinomial Bayes Classifier, and a Support Vector Classifier. The Support Vector Classifier had the highest accuracy (0.86) and the built in F1 score (0.93) out of the three. However all models misclassified over 50% of the minority class tweets. 

In addition to upsampling, we also tried eliminating some of the most frequent features shared by both classes such as "Google", "SXSW", and "Apple." The Gaussian Naïve Bayes lowered its misclassification of Negative Tweets; however, it had poorer performance for the positive tweets. The SVC showed no improvement since it eliminates features automatically.

We then made this problem a multiclass problem by including over 5,000 neutral tweets to see how the model would perform. For this portion, we only used Gaussian Naïve Bayes and Multinomial Bayes Classifiers because SVC took too long to train. A resampled Multinomial Bayes Classifer performed the best out of the two with an f1 score of 0.51.
![Image](images/naive_resample.png?raw=true)
![Image](images/unsample_multi.png?raw=true)
![Image](images/resample_multi.png?raw=true)

## Conclusions
The final predictive model is ultimately to specific to both its historical and cultural context. Terms used to describe positive and negative reactions can be somewhat dated, i.e. "#humblebrag," "win," and "hipster" as positive reactions. 
The emotional classification was also somewhat dubious. What were classified as negative reactions towards iPhone 3G's battery life were also negative reactions to apps that drained battery life and to the festival in general for not having enough places to charge. Provocative opinions from panelists can have a lot of longevity on twitter.
The model functions as more of proof concept for something that could be integrated in an ETL-pipe. In the future, we would like to employ a historic word2vec for sentiment analysis and for eventually scaling our model. We would like improve cleaning and preprocessing steps by integrating more custom spaCy extensions and attributes.
To correct the dubious emotional classifications, it would be worthwhile to employ unsupervised models like hierarchical clustering in order to find natural emotional clusters/market segmentation as a means of comparision. In addition, considering how eliminating common features improved classification of negative reaction tweets it would be interesting to train some models on frequency counts rather than TF-IDF to compare performance and improve feature engineering.

