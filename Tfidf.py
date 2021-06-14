# -*- coding: utf-8 -*-
"""
TF-IDF
"""

import nltk

paragraph= """New York:Scientists have found how misfiring of the immune system in some patients infected with the novel coronavirus can lead to severe consequences of COVID-19, an advance which may help identify those at high risk of dying from the disease, and suggest drugs to treat them.

The researchers from Yale University in the US examined 113 patients admitted to the Yale New Haven Hospital, and analysed the varying immune system responses they exhibited during their hospital stay, from admittance to discharge, or death.

The study, published in the journal Nature, found that all patients shared a common COVID-19 "signature" in immune system activity early in the course of disease.

However, those who experienced only moderate symptoms exhibited diminishing immune system responses and viral particle levels in their bodies over time, the researchers said.

"This study shows how people's immune system responds to SARS-CoV-2. It shows that wrong types of immune responses are engaged in severe cases and how some of these are associated with mortality," study senior author Akiko Iwasaki from Yale University noted in a tweet.

According to the scientists, patients who went on to develop severe cases of the disease showed no decrease in the levels of the virus particles in their body, or immune system reaction.

They said many of the immune signals in these patients, including the molecules called cytokines, were accelerated.

But even in the early course of treatment, the researchers found factors which predicted which patients were at greatest risk of developing severe forms of the illness.

"We were able to pull out signatures of disease risk," Akiko Iwasaki said.

While earlier studies have identified that the immune system unleashed a massive and damaging "cytokine storm" in severe cases of COVID-19, the scientists said the specific elements of this response were unknown.

The current study found that one risk factor was the presence of an immune system molecule called alpha interferon - a cytokine mobilised to combat viral pathogens such as the flu virus.

However, the researchers said COVID-19 patients with high levels of alpha interferon fared worse than those with low levels.

"This virus just doesn't seem to care about alpha interferon. The cytokine appears to be hurting, not helping," Akiko Iwasaki said.

According to the study, another early prognosticator of poor outcomes is activation of a complex of proteins that detects pathogens and triggers an inflammatory response to infection called inflammasome.

The researchers said inflammasome activation was linked to poor outcomes and death in several patients.

They found that people who respond better to the infection tend to express high levels of growth factors - a type of immune system molecules that repairs tissue damage to the linings of blood vessels and lungs.

Taken together, the researchers believe the data can help predict patients at high risk of poor outcomes from the disease."""


import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#text cleaning
wordnet = WordNetLemmatizer()

sentences = nltk.sent_tokenize(paragraph)
corpus=[]

for i in range(len(sentences)):
    preProcess = re.sub('[^a-zA-Z]',' ',sentences[i])
    preProcess = preProcess.lower()
    preProcess = preProcess.split()
    preProcess = [wordnet.lemmatize (word) for word in preProcess if  not word in set(stopwords.words('english'))]
    preProcess = ' '.join(preProcess)
    corpus.append(preProcess)
    
#TF-IDF
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus).toarray()
    
    
    
