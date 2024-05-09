#IMPORTING REQUIRED MODULES

from flask import Flask, render_template, request, redirect, session, url_for,jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import re
import pickle
import requests
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv
import pandas as pd

#INITIALIZING FLASK
app = Flask(__name__)

# Function to fetch movie posters
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=3d158574e427b05b2c016e3b8ac81f4f&language=en-US".format(movie_id)
    data = requests.get(url).json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path, data['overview']




# Function to recommend movies based on rating range
def recommend_rating(movies):
    movies = movies[(movies['vote_average'] <= 10)]
    if movies.empty:
        return "No movies found for the provided rating range."
    return movies.head(10)  # Return only top 10 recommendations



# Function to recommend movies based on popularity range
def recommend_popularity(movies1):
    movies1 = movies1[(movies1['popularity'] <= 12000)]  # Adjust the popularity threshold as needed
    if movies1.empty:
        return "No movies found for the provided popularity range."
    return movies1.head(10)  # Return only top 10 recommendations



# Load movie data
movies = pickle.load(open("genre_list.pkl", 'rb'))
movies1 = pickle.load(open("pop_list.pkl", 'rb'))


# -------------------------------------------------------------------------------------------------------------------
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:root@localhost/WikiCinema"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Account(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    password = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), nullable=False)


#LOGIN PAGE
@app.route('/', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        account = Account.query.filter_by(username=username, password=password).first()
        if account:
            session['loggedin'] = True
            session['id'] = account.id
            session['username'] = account.username
            msg = 'Logged in successfully !'
            return redirect(url_for('home'))
        else:
            msg = 'Incorrect username / password !'
    return render_template('login.html', msg=msg)


@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))


#REGISTER
@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        existing_account = Account.query.filter_by(username=username).first()
        if existing_account:
            msg = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address !'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers !'
        elif not username or not password or not email:
            msg = 'Please fill out the form !'
        else:
            new_account = Account(username=username, password=password, email=email)
            db.session.add(new_account)
            db.session.commit()
            msg = 'You have successfully registered !'
            return render_template('login.html', msg=msg)
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('register.html', msg=msg)


#HOME PAGE
@app.route('/home')
def home():
    rating_movies = recommend_rating(movies)
    popularity_movies = recommend_popularity(movies1)
    rating_movies_data = []
    popularity_movies_data = []

    # Fetching data for movies based on rating range
    if not isinstance(rating_movies, str):
        for i, movie in rating_movies.iterrows():
            poster_url, overview= fetch_poster(movie['id'])
            rating_movies_data.append({
                'title': movie['title'],
                'poster_url': poster_url,
                'overview': overview
            })

    # Fetching data for movies based on popularity range
    if not isinstance(popularity_movies, str):
        for i, movie in popularity_movies.iterrows():
            poster_url, overview= fetch_poster(movie['id'])
            popularity_movies_data.append({
                'title': movie['title'],
                'poster_url': poster_url,
                'overview': overview
            })

    return render_template('index.html', rating_movies=rating_movies_data, popularity_movies=popularity_movies_data)




# -------------------------------------------------------------------------------------------------------------------
movies2 = pickle.load(open("movies_list.pkl", 'rb'))
similarity = pickle.load(open("similarity.pkl", 'rb'))

#SEARCH---------------------------------------------

def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

movies["clean_title"] = movies["title"].apply(clean_title)

# Create TF-IDF vectorizer (Term Frequency - Inverse Document Frequency)
# Converts the search title into numbers that the computer can understand and use to recommend movies
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])

# Movie search function
def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    return results

#---------------------------------------------------------------

@app.route('/recommendation')
def recommender():
    movies = pickle.load(open("movies_list.pkl", 'rb'))
    movies_list = movies['title'].values.tolist()
    return render_template('recommend.html', movies_list=movies_list)

@app.route('/recommendations', methods=['POST'])
def recommendations():

    movie_name = request.form['movie']
    search_results = search(movie_name)

    if not search_results.empty:
            
            movie_name = search_results.iloc[0]['title']
            index = movies2[movies2['title'] == movie_name].index[0]
            distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector:vector[1])
            recommend_movie = []
            recommend_poster = []
            for i in distance[1:11]:
                movies_id = movies2.iloc[i[0]].id
                recommend_movie.append(movies2.iloc[i[0]].title)
                recommend_poster.append(fetch_poster(movies_id))
            
            return jsonify({'movie_name': movie_name, 'recommend_movie': recommend_movie, 'recommend_poster': recommend_poster})
    else:
            
            return jsonify({'error': 'No movie found matching the search criteria.'})

#------------------------------------------------------------------------------------------------------------------
#GENRE BASED RECOMMENDATION
def recommend_genre_and_rating(movies, genre, min_rating, max_rating):


    input_genres = genre.split(',')
    genre_movies = movies[movies['genre'].notnull() & movies['genre'].apply(lambda x: isinstance(x, str) and any(g in x.split(',') for g in input_genres))]
    if genre_movies.empty:
        return "No movies found for the provided genre(s)."
    genre_movies = genre_movies[(genre_movies['vote_average'] >= min_rating) & (genre_movies['vote_average'] <= max_rating)]
    if genre_movies.empty:
        return "No movies found for the provided genre(s) and rating range."
    vectorizer = CountVectorizer()
    overview_vectors = vectorizer.fit_transform(genre_movies['overview'])
    similarity_matrix = cosine_similarity(overview_vectors)
    top_rated_movies = genre_movies.sort_values(by='vote_average', ascending=False)
    return top_rated_movies.head(10), similarity_matrix


@app.route('/genre')
def genre():
    return render_template('search.html')


@app.route('/results', methods=['POST'])
def results():
    movies = pickle.load(open("genre_list.pkl", 'rb'))
    genre_input = request.form['genre']
    min_rating = float(request.form['min_rating'])
    max_rating = float(request.form['max_rating'])
    if min_rating < max_rating:
        recommended_movies, _ = recommend_genre_and_rating(movies, genre_input, min_rating, max_rating)
        if isinstance(recommended_movies, str):
            return jsonify({'message': recommended_movies})
        else:
            recommendations = []
            for _, movie in recommended_movies.iterrows():
                recommendations.append({
                    'title': movie['title'],
                    'poster': fetch_poster(movie['id']),
                    'rating': movie['vote_average']
                })
            return jsonify({'movies': recommendations})
    
    else:
        rating_msg = {"message": "minimum rating should be less than maximum rating!"}
        return jsonify(rating_msg)


#----------------------------------------------------------------------------------------------------------------
#POPULARITY BASED RECOMMENDATION


def recommend_genre_and_popularity(movies, genre, min_rating, max_rating):
    input_genres = genre.split(',')
    genre_movies = movies[movies['genre'].notnull() & movies['genre'].apply(lambda x: isinstance(x, str) and any(g in x.split(',') for g in input_genres))]
    if genre_movies.empty:
        return "No movies found for the provided genre(s)."
    genre_movies = genre_movies[(genre_movies['popularity'] >= min_rating) & (genre_movies['popularity'] <= max_rating)]
    if genre_movies.empty:
        return "No movies found for the provided genre(s) and rating range."
    vectorizer = CountVectorizer()
    overview_vectors = vectorizer.fit_transform(genre_movies['overview'])
    similarity_matrix = cosine_similarity(overview_vectors)
    top_rated_movies = genre_movies.sort_values(by='popularity', ascending=False)
    return top_rated_movies.head(10), similarity_matrix


#FILTER BY POPULARITY
@app.route('/popularity')
def popularity():
    return render_template('search.html')


#FETCHING RESULTS
@app.route('/results_popu', methods=['POST'])
def results_popu():
    movies = pickle.load(open("pop_list.pkl", 'rb'))
    genre_input = request.form['genre']
    min_rating = float(request.form['min_rating'])*1100
    max_rating = float(request.form['max_rating'])*1100
    recommended_movies, _ = recommend_genre_and_popularity(movies, genre_input, min_rating, max_rating)
    if isinstance(recommended_movies, str):
        return jsonify({'message': recommended_movies})
    else:
        recommendations = []
        for _, movie in recommended_movies.iterrows():
            recommendations.append({
                'title': movie['title'],
                'poster': fetch_poster(movie['id']),
                'rating': movie['popularity']
            })
        return jsonify({'movies': recommendations})
    
#----------------------------------------------------------------------------------------------------------------

@app.route('/filter')
def filter():
    return render_template("filter.html")

#CREDITS---------------------------------------------------------------

@app.route('/credits')
def credits():
    return render_template("credits.html")

#----------------------------------------------------------------------------------------------------------------



#ADD MOVIES---------------------------------------------------------------

# Function to update dataset and pickle files
def update_dataset(data):
    # Write data to CSV file
    with open('dataset.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(data)

    # Calculate similarity and update pickle files
    calculate_and_update_pickle_files()

# Function to calculate similarity and update pickle files
def calculate_and_update_pickle_files():
    cv = CountVectorizer(max_features=10000, stop_words='english') 
    movies = pd.read_csv('dataset.csv')
    movies1 = movies[['id','title','overview','genre']]
    movies1['tags'] = movies1['overview'] + ' ' + movies1['genre']  # Combine overview and genre
    new_data = movies1.drop(columns=['overview', 'genre'])
    vector = cv.fit_transform(new_data['tags'].values.astype('U')).toarray()
    similarity = cosine_similarity(vector)
    movies2 = movies[['id','title','overview','genre','vote_average','popularity']].sort_values(by='popularity', ascending=False)
    movies3 = movies[['id','title','overview','genre','vote_average','popularity']].sort_values(by='vote_average', ascending=False)

    # Update pickle files
    pickle.dump(similarity, open('similarity.pkl', 'wb'))
    pickle.dump(movies3, open('genre_list.pkl', 'wb'))
    pickle.dump(new_data, open('movies_list.pkl', 'wb'))
    pickle.dump(movies2, open('pop_list.pkl', 'wb'))



# Define route for homepage
@app.route('/Add')
def add():
    return render_template('add_movies.html')


# Define route to handle form submission
@app.route('/add_movie', methods=['POST'])
def add_movie():
    if request.method == 'POST':
        # Get data from form
        id = request.form['id']
        title = request.form['title']
        genre = request.form['genre']
        original_language = request.form['original_language']
        overview = request.form['overview']
        popularity = request.form['popularity']
        release_date = request.form['release_date']
        vote_average = request.form['vote_average']
        vote_count = request.form['vote_count']

        # Convert data types if necessary
        popularity = float(popularity)
        vote_average = float(vote_average)
        vote_count = int(vote_count)

        # Create data row
        data = [id, title, genre, original_language, overview, popularity, release_date, vote_average, vote_count]

        # Update dataset and pickle files
        update_dataset(data)

        return render_template('add_movies.html', message="Movie added successfully!")
    return render_template('add_movies.html')


#----------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    app.secret_key = 'Hello@123'  # You should set your secret key for session management
    app.run(debug=True)
