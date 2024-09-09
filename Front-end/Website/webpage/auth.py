from flask import Blueprint

auth = Blueprint('auth', __name__)

@auth.route('/login')
def login():
    return "Hello"

@auth.route('/logout')
def logout():
    return "Logout"

@auth.route('/signin')
def signin():
    return "signin"



