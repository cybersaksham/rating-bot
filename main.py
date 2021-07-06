import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier

# Importing data
data = pd.read_csv("reviews.csv")

# Extracting features
X = data["Reviews"]
vec = CountVectorizer()
vec.fit(X)
vec_x = vec.transform(X)

# Tf Idf extraction
tfidf = TfidfTransformer()
tfidf.fit(vec_x)
rev = tfidf.transform(vec_x)

# Y-axis
Y = data["Rating"].tolist()

# Rating Model
Model = DecisionTreeClassifier()
Model.fit(rev, Y)

# Working
# txt = ["The product is not in good condition", ]
txt = input("Enter a review (Q for quit) : ")
while txt != "Q":
    txt_ex = vec.transform([txt, ])
    txt_tf = tfidf.transform(txt_ex)
    print(Model.predict(txt_tf)[0])
    txt = input("Enter a review (Q for quit) : ")
