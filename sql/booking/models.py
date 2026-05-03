import os

from dotenv import load_dotenv
from sqlalchemy import Column, Integer, String, create_engine, DateTime
from sqlalchemy.orm import declarative_base

load_dotenv()

#"mysql+pymysql://root:123456@localhost:3306/test_db"
DATABASE_URL = (
    f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}"
    f"@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}"
    f"/{os.getenv('MYSQL_DATABASE')}"
)

engine = create_engine(
    DATABASE_URL,
    echo=True
)
# 1.定义表
Base = declarative_base()
class Venue(Base):
    __tablename__ = "venues"
    id = Column(Integer,primary_key=True,autoincrement=True)
    name=Column(String(100),unique=True,nullable=False)
    type=Column(String(50))
    location=Column(String(200))

class Booking(Base):
    __tablename__ = "bookings"
    id = Column(Integer,primary_key=True,autoincrement=True)
    venue_id=Column(Integer)
    student_id=Column(Integer)
    start_time=Column(DateTime)
    end_time=Column(DateTime)
    status = Column(String(20), nullable=False, default="pending")

#3.创建表
print("Tables to create:", Base.metadata.tables.keys())
Base.metadata.create_all(engine)
