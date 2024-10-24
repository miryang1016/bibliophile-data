import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

path = './book_noun.xlsx'
data = pd.read_excel(path)
df = pd.DataFrame(data[['title_nm', 'nouns']])

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['nouns'])

title_to_index = dict(zip(df['title_nm'], df.index))

def content_recommendations(title, request_number):
    idx = title_to_index.get(title)
    if idx is None:
        return "해당 제목이 데이터에 없습니다."

    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    start = 50 + 10 * request_number
    sim_scores = sim_scores[start+1:start+10]
    book_indices = [idx[0] for idx in sim_scores]

    unique_indices = list(dict.fromkeys(book_indices))
    recommendations = df[['title_nm']].iloc[unique_indices]

    return recommendations['title_nm'].tolist()