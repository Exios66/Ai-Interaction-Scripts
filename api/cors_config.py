CORS_SETTINGS = {
    'origins': [
        'https://yourusername.github.io',  # Replace with your GitHub Pages domain
        'http://localhost:5000',
        'http://127.0.0.1:5000'
    ],
    'methods': ['GET', 'POST', 'OPTIONS'],
    'allow_headers': ['Content-Type', 'Authorization'],
    'expose_headers': ['Content-Range', 'X-Total-Count'],
    'supports_credentials': True,
    'max_age': 600
} 