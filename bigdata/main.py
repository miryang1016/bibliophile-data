from fastapi import FastAPI, Response, HTTPException, Query, Depends
import pandas as pd
from pydantic import BaseModel
from konlpy.tag import Kkma
from typing import List
from wc import generate_wordcloud
from content import content_recommendations
from collaborative import collaborative_recommendations
from tag import tag_recommendations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from data import User, Interest, Review
from collections import defaultdict

DATABASE_URL = ""

# 데이터베이스 연결 및 세션 설정
engine = create_engine(DATABASE_URL, pool_recycle=500, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = SessionLocal()

app = FastAPI()

# 의존성으로 사용할 세션 생성 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 전체 책 데이터
all_path = './book.xlsx'
all_data = pd.read_excel(all_path)
all_df = pd.DataFrame(all_data[['title_nm', 'kdc', 'books_id']])

@app.get("/recommend")
async def root():
    return {"message": "FastAPI 서버가 정상적으로 실행 중입니다!"}

class ContentData(BaseModel):
    content: list

@app.post("/recommend/wordcloud")
async def wordcloud_view(request_data: ContentData):
    
    df = pd.DataFrame(request_data.dict())
    df['content'] = df['content'].str.replace('[^가-힣]', ' ', regex=True) # 정규표현식

    kkma = Kkma()
    nouns = df['content'].apply(kkma.nouns).explode()
    
    if nouns.empty:
        
        return {"message": "No valid nouns found in the input."}
    
    df_word = pd.DataFrame({'word': nouns})
    df_word['count'] = df_word.groupby('word')['word'].transform('count')
    df_word = df_word[df_word['word'].str.len() >= 2].drop_duplicates().sort_values('count', ascending=False)

    dic_word = df_word.set_index('word')['count'].to_dict()
    img = generate_wordcloud(dic_word)

    return Response(content=img.read(), media_type="image/png")


@app.post("/recommend/content")
async def get_content(title: str, request_number: int = Query(1, ge=0)):
    if not title:
        raise HTTPException(status_code=400, detail="책 제목이 필요합니다.")
    
    recommendations = content_recommendations(title, request_number)
    
    if isinstance(recommendations, str):
        raise HTTPException(status_code=404, detail=recommendations)
    
    return {"title": title, "recommendations": recommendations}

# 전체 유저 정보 생성

user_data = defaultdict(lambda: {"ratings": {}})

async def get_user(db: Session = Depends(get_db)):

    query = db.query(
        User.user_id,
        Review.book_id,
        Review.star
    ).join(Review, User.user_id == Review.user_id) \
    .all()
    
    if not query:
        return {"message": "No data found"}

    global user_data
    for user_id, book_id, star in query:

        user_data[str(user_id)]["ratings"][f"Book{book_id}"] = star

    return user_data

async def get_interest(user_id: int, db: Session):
    interests = db.query(Interest).filter(Interest.user_id == user_id).all()
    return [interest.classification for interest in interests] 
      
@app.post("/recommend/collaborative")
async def get_collaborative(id: int, db: Session = Depends(get_db)):
        
    user_data = await get_user(db)
    
    user = db.query(User).filter(User.user_id == id).first()
    if not user:
        raise HTTPException(status_code=404, detail="유저를 찾을 수 없습니다.")

    ratings_list = []
    for user_id, user_info in user_data.items():
        ratings = user_info['ratings']
        for book_id, rating in ratings.items():
            ratings_list.append({"user_id": user_id, "book_id": book_id, "rating": rating})
    
    ratings_df = pd.DataFrame(ratings_list)
    pivot_df = ratings_df.pivot(index='user_id', columns='book_id', values='rating')
    user_interest_numbers = await get_interest(id, db)
    recommendations = await collaborative_recommendations(id, user_data, user_interest_numbers, all_df, pivot_df.reset_index())
    
    return {"user_id": id, "recommended_books": recommendations}


class TagRequest(BaseModel):
    id: int
    tags: List[str]

@app.post("/recommend/tag")
async def get_tag(request: TagRequest, db: Session = Depends(get_db)):
    
    user_data = await get_user(db)
    
    user = db.query(User).filter(User.user_id == request.id).first()
    if not user:
        raise HTTPException(status_code=404, detail="유저를 찾을 수 없습니다.")

    ratings_list = []
    for user_id, user_info in user_data.items():
        ratings = user_info['ratings']
        for book_id, rating in ratings.items():
            ratings_list.append({"user_id": user_id, "book_id": book_id, "rating": rating})
    
    ratings_df = pd.DataFrame(ratings_list)
    pivot_df = ratings_df.pivot(index='user_id', columns='book_id', values='rating')
    
    recommendations = await tag_recommendations(request.id, user_data, request.tags, all_df, pivot_df.reset_index())

    return {"user_name": request.id, "recommendations": recommendations}
