import os
import json
import threading
from functools import lru_cache
import re
import pandas as pd
import numpy as np
import whois
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from sklearn.base import BaseEstimator, TransformerMixin
import concurrent.futures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def load_cache(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache, filename):
    with open(filename, 'w') as f:
        json.dump(cache, f)

cache_file = os.path.join(CACHE_DIR, 'feature_cache.json')
cache_lock = threading.Lock()
feature_cache = load_cache(cache_file)

@lru_cache(maxsize=None)
def get_cached_feature(url):
    with cache_lock:
        return feature_cache.get(url)

def set_cached_feature(url, features):
    with cache_lock:
        feature_cache[url] = features
        save_cache(feature_cache, cache_file)

def add_scheme(url):
    if not urlparse(url).scheme:
        return 'http://' + url
    return url

class ArgumentationFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, extract_domain_age=True, extract_ssl_info=True, extract_alexa_rank=True):
        self.extract_domain_age = extract_domain_age
        self.extract_ssl_info = extract_ssl_info
        self.extract_alexa_rank = extract_alexa_rank
    
    def get_domain_age(self, domain):
        try:
            whois_info = whois.whois(domain)
            creation_date = whois_info.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            domain_age = (pd.Timestamp.now() - pd.Timestamp(creation_date)).days
            return domain_age
        except:
            return -1  # If WHOIS lookup fails

    def get_ssl_info(self, url):
        try:
            session = requests.Session()
            retry = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            response = session.get(url, timeout=10)
            ssl_info = response.url.startswith('https')
            return 1 if ssl_info else 0
        except requests.exceptions.RequestException:
            return -1  # If SSL lookup fails

    def get_alexa_rank(self, domain):
        try:
            url = f"http://data.alexa.com/data?cli=10&dat=s&url={domain}"
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'xml')
            rank = soup.find('REACH')['RANK']
            return int(rank)
        except requests.exceptions.RequestException:
            return -1  # If Alexa rank lookup fails

    def get_features(self, url):
        cached_features = get_cached_feature(url)
        if cached_features:
            return cached_features
        
        features = {}
        url = add_scheme(url)
        parsed_url = urlparse(url)
        domain = parsed_url.netloc

        features['url_length'] = len(url)
        features['dot_count'] = url.count('.')
        features['has_at_symbol'] = '@' in url
        features['subdomain_count'] = len(domain.split('.')) - 2
        features['has_ip'] = bool(re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', domain))
        features['https'] = 1 if parsed_url.scheme == 'https' else 0
        features['suspicious_chars'] = len(re.findall(r'[?=&\-]', url))
        
        if self.extract_domain_age:
            features['domain_age'] = self.get_domain_age(domain)
        if self.extract_ssl_info:
            features['ssl_valid'] = self.get_ssl_info(url)
        if self.extract_alexa_rank:
            features['alexa_rank'] = self.get_alexa_rank(domain)
        
        set_cached_feature(url, features)
        return features
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_series = pd.Series(X)
        X_series = X_series.apply(add_scheme)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            features_list = list(executor.map(self.get_features, X_series))
        return pd.DataFrame(features_list)

# Load the dataset
data = pd.read_csv('phishing__data.csv')

X = data['url']  # Only the URLs
y = data['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

extractor_args = {
    'extract_domain_age': True,
    'extract_ssl_info': True,
    'extract_alexa_rank': True
}

pipeline_rf = Pipeline([
    ('feature_extraction', ArgumentationFeatureExtractor(**extractor_args)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline_rf.fit(X_train, y_train)

rf_y_pred = pipeline_rf.predict(X_test)

st.title('Phishing URL Detection')
url_input = st.text_input('Enter URL for prediction', '')

if url_input:
    prediction = pipeline_rf.predict([url_input])
    if prediction[0] == 1:
        st.write(f"{url_input} is predicted as PHISHING.")
    else:
        st.write(f"{url_input} is predicted as NOT PHISHING.")
