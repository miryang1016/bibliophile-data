from sqlalchemy import Column, BigInteger, String, Enum, Integer, ForeignKey, DateTime, DECIMAL, Date
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = 'user'

    user_id = Column(BigInteger, primary_key=True, autoincrement=True)
    birthday = Column(Date)
    created_date = Column(DateTime(6))
    last_modify_date = Column(DateTime(6))
    email = Column(String(255))
    nickname = Column(String(255))
    profile_image_url = Column(String(255))
    word_cloud_img_url = Column(String(255))
    gender = Column(Enum('MAN', 'WOMAN'))
    oauth_server_type = Column(Enum('GOOGLE', 'KAKAO', 'NAVER'))

class Interest(Base):
    __tablename__ = 'interest'

    created_date = Column(DateTime(6), nullable=True)
    interest_id = Column(BigInteger, primary_key=True, autoincrement=True)
    last_modify_date = Column(DateTime(6), nullable=True)
    user_id = Column(BigInteger, ForeignKey('user.user_id'), nullable=True)
    classification = Column(Enum('SOCIETY', 'FICTION', 'SCIENCE', 'ARTS', 'LANGUAGE', 'HISTORY'), nullable=True)

class Review(Base):
    __tablename__ = 'review'

    star = Column(Integer, primary_key=True)
    book_id = Column(BigInteger, ForeignKey('book.book_id'), nullable=True)
    created_date = Column(DateTime(6), nullable=True)
    last_modify_date = Column(DateTime(6), nullable=True)
    review_id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('user.user_id'), nullable=True)
    content = Column(String(255), nullable=True)
