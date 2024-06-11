from flask import Flask, jsonify, request
from processChat import processRequest
from webCrawl import crawl
from buildEmbeddings import build_embeddings
from customQnA import answer_question, read_embeddings
import argparse
from urllib.parse import urlparse
import os

app = Flask(__name__)

# Sample data (you can replace this with your own data or database connection)
chatRequest = [
    {'id': 1, 'request': 'Description for Task 1'},
    {'id': 2, 'request': 'Description for Task 2'}
]

parser = argparse.ArgumentParser()
parser.add_argument('--url', type=str, help='A required full url including https://',)
parser.add_argument('--clean', type=bool, help='Clean the url directory if already exist')

args = parser.parse_args()

full_url = ""
clean_crawl = False

if os.environ.get('FULL_URL'):
    full_url = os.environ['FULL_URL']
    print('FULL_URL: ', full_url)

if os.environ.get('CLEAN'):
    clean_crawl = os.environ['CLEAN']
    print('CLEAN: ', clean_crawl)

if args.url is not None:
    full_url = args.url
else:
    print("args.url is not provided")

if args.clean is not None:
    clean_crawl = args.clean
else:
    print("args.clean is not provided")

if full_url == "":
    print("Please provide a valid url")
    exit()

domain = ""
if "https://" in full_url:
    domain = urlparse(full_url).netloc

if domain == "":
    print("Please provide a valid url")
    exit()

url_dir = "urls/"
text_dir = "text.url/"
process_dir = "processed.url/"

if not os.path.exists(process_dir+domain) or clean_crawl == True:
    print('crawling url: ', full_url)
    crawl(full_url, text_dir=text_dir, url_dir=url_dir )
    print('building embeddings in: ', process_dir+domain)
    build_embeddings(domain, text_dir=text_dir, url_dir=url_dir, process_dir=process_dir)

# Route to get all tasks
@app.route('/chatbot', methods=['GET'])
def get_tasks():
    return jsonify({'Request': chatRequest})

# Route to create a new task
@app.route('/chatbot', methods=['POST'])
def create_task():
    question = request.json.get('request', '')
    new_request = {
        'id': request.json['id'],
        'request': question,
    }
    df = read_embeddings(process_dir="processed.url/")
    print(question)
    answer = answer_question(df, question=question)
    print(answer)
    return jsonify({
        'id': request.json['id'],
        'Response':answer,
    })
    # return jsonify({'Request': new_request}), 201

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
