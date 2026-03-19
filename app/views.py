from flask import Blueprint, render_template

views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@views.route('/test', methods=['GET', 'POST'])
def test():
    return render_template('index.html')