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


async def tag_recommendations(id, users_data, user_interest, all_df, dataframe):
    
    user_data = users_data.get(str(id))
    kdc_list = map_interests_to_numbers(user_interest)
    num_categories = len(kdc_list)
    print(num_categories)
    
    if user_data is None:
        return recommendations_norated(all_df[all_df['kdc'].isin(kdc_list)])
    else:
        rated_books = list(user_data['ratings'].keys())

    # 평가한 책이 있는 경우
    user_df = (dataframe[dataframe['user_id'] == str(id)][rated_books]).dropna()
    user_df = user_df.select_dtypes(include=[np.number])

    other_users_df = dataframe.loc[:, rated_books].drop(dataframe[dataframe['user_id'] == id].index)
    other_users_df = other_users_df.fillna(0)

    if other_users_df.empty:
        return recommendations_norated(all_df[all_df['kdc'].isin(kdc_list)])

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

    # 예상 평점이 높은 책
    top_books = sorted(recommender.items(), key=lambda x: x[1], reverse=True)
    top_book_ids = convert_recommendations([book[0] for book in top_books])
    filtered_books = all_df[all_df['kdc'].isin(kdc_list)]
    
    filtered_ids = filtered_books['books_id'].tolist()
    top_in_list = [book for book in top_book_ids if book in filtered_ids]
    print(top_in_list)
    top_3_books = top_in_list[:3]
    
    # 관심사에 따른 3권 랜덤 추천
    books_needed = 6 - len(top_3_books)
    books_per_category = books_needed // num_categories
    remainder_books = books_needed % num_categories

    # 이미 평가된 책은 제외
    filtered_books = filtered_books[~filtered_books['books_id'].isin(rated_books)]

    random_3_books = []

    for i, kdc in enumerate(kdc_list):
        count = books_per_category

        # 부족한 책 수만큼 추가
        if i < remainder_books:
            count += 1
        
        category_books = filtered_books[filtered_books['kdc'] == kdc].sample(n=min(count, len(filtered_books[filtered_books['kdc'] == kdc])), replace=False)
        
        random_3_books.extend(category_books['books_id'].tolist())

        # 최종 추천 리스트: 3권 예상 평점 높은 책 + 3권 관심사 기반 랜덤 책(default)
        final_recommendations = list(top_3_books) + list(random_3_books)

        if len(random_3_books) < books_needed:
            additional_books_needed = books_needed - len(random_3_books)
            additional_books = filtered_books.sample(n=additional_books_needed, replace=False)['books_id'].tolist()
            final_recommendations.extend(additional_books)

    return convert_recommendations(final_recommendations)
