import numpy as np
import pandas as pd
import re
import random
import operator
from sklearn.metrics.pairwise import cosine_similarity

# 관심사 처리
interest_mapping = {
    "SOCIETY": [3],
    "HISTORY": [9],
    "ARTS": [6],
    "FICTION": [8],
    "SCIENCE": [4],
    "LANGUAGE": [7]
}

def map_interests_to_numbers(interest_list):
    unique_numbers = set()
    for interest in interest_list:
        if interest in interest_mapping:
            unique_numbers.update(interest_mapping[interest])
    return list(unique_numbers)  # 리스트로 변환


def recommendations_norated(filtered_books):
    recommended_books = filtered_books.sample(n=min(6, len(filtered_books)))
    final_recommendations = list(recommended_books['books_id'])
    random.shuffle(final_recommendations)
    return final_recommendations


def cosine_similarity_func(user_vector, other_users_df):
    sim_dict = {}
    
    for other_user in other_users_df.index:
        other_user_vector = other_users_df.loc[other_user].values.reshape(1, -1)
        sim = cosine_similarity(user_vector, other_user_vector)[0, 0]
        sim_dict[other_user] = sim
    return sorted(sim_dict.items(), key=operator.itemgetter(1), reverse=True)


def convert_recommendations(recommendations):
    converted_recommendations = []
    
    for item in recommendations:
        if isinstance(item, str):
            if item.startswith("Book"):
                number = re.findall(r'\d+', item)
                if number:
                    converted_recommendations.append(int(number[0]))
            continue
        else:
            converted_recommendations.append(int(item))
    
    return converted_recommendations


async def collaborative_recommendations(id, users_data, user_interest, all_df, dataframe):
    
    user_data = users_data.get(str(id))
    user_interest_numbers = map_interests_to_numbers(user_interest)
    
    if user_data is None:
        return recommendations_norated(all_df[all_df['kdc'].isin(user_interest_numbers)])
    else:
        rated_books = list(user_data['ratings'].keys())

    # 평가한 책이 있는 경우
    user_df = (dataframe[dataframe['user_id'] == str(id)][rated_books]).dropna()
    user_df = user_df.select_dtypes(include=[np.number])

    other_users_df = dataframe.loc[:, rated_books].drop(dataframe[dataframe['user_id'] == id].index)
    other_users_df = other_users_df.fillna(0)


    if other_users_df.empty:
        return recommendations_norated(all_df[all_df['kdc'].isin(user_interest_numbers)])

    sim_mat = cosine_similarity_func(user_df.values.reshape(1, -1), other_users_df)

    recommend_list = list(set(dataframe.columns) - set(rated_books))
    others_k = [i[0] for i in sim_mat if i[1] > 0]
    recommender = {}
    
    for book in recommend_list:
        ratings = []
        sims = []
        for other_user in others_k:
            user_ratings = dataframe[dataframe['user_id'] == other_user][book]
            if not user_ratings.empty and not pd.isna(user_ratings.values[0]):
                ratings.append(user_ratings.values[0])
                sims.append(dict(sim_mat)[other_user])
                
        if len(sims) >= int(len(sims) / 10):  # 전체 인원의 10%
            pred = np.dot(sims, ratings) / sum(sims) if sum(sims) != 0 else 0
            recommender[book] = int(pred)

    # 예상 평점이 높은 책들 중에서 4권 선택
    top_4_books = sorted(recommender.items(), key=lambda x: x[1], reverse=True)[:4]
    top_4_books = [(book, rating) for book, rating in top_4_books if book != 'user_id']
    print(top_4_books)
    
    # 관심사에 따른 2권 랜덤 추천
    filtered_books = all_df[all_df['kdc'].isin(user_interest_numbers)]
    print(user_interest_numbers)
    print(filtered_books)
    filtered_books = filtered_books[~filtered_books['books_id'].isin(rated_books)]
    random_2_books = filtered_books.sample(n=6 - len(top_4_books))

    # 최종 추천 리스트: 4권 예상 평점 높은 책 + 2권 관심사 기반 랜덤 책(default)
    final_recommendations = [book for book, _ in top_4_books] + list(random_2_books['books_id'])

    return convert_recommendations(final_recommendations)
