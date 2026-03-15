# ======================================================
# NEWS CATEGORY DATASET UTILITY
# ======================================================
# This file provides a simple news dataset if you want
# to focus specifically on common news categories.
# ======================================================

def get_news_categories():
    return [
        "Business", "Tech", "Politics", "Sports", "Entertainment", "Health", "Science"
    ]

def get_sample_news_data():
    """Returns a dictionary mapping text to categories"""
    return [
        ("Apple announces new iPhone with advanced AI features", "Tech"),
        ("The stock market hit a record high today as inflation cools", "Business"),
        ("Congress votes on a new infrastructure bill to repair bridges", "Politics"),
        ("Lionel Messi scores a hat-trick in the latest Inter Miami match", "Sports"),
        ("The new superhero movie broke box office records this weekend", "Entertainment"),
        ("FDA approves a new drug that could significantly reduce heart disease", "Health"),
        ("NASA's Webb telescope discovers a galaxy from the early universe", "Science"),
        ("Microsoft acquires new gaming studio to expand Game Pass library", "Tech"),
        ("Oil prices drop as global demand weakens in the second quarter", "Business"),
        ("The upcoming election is expected to have a record voter turnout", "Politics"),
        ("The Olympics closing ceremony featured spectacular firework displays", "Sports"),
        ("Grammy awards celebrate the best artists and albums of the year", "Entertainment"),
        ("New study shows that exercise can improve mental health and longevity", "Health"),
        ("Scientists find evidence of ancient liquid water on Mars' surface", "Science")
    ]
