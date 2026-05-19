from dataclasses import dataclass

@dataclass
class User:
    user_id: int
    city: str
    region: str
    country: str

@dataclass
class Book:
    isbn: str
    title: str
    author: str
    year_of_publication: str
    publisher: str

@dataclass
class Rating:
    user_id: int
    isbn: str
    rating: int