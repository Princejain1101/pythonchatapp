from flask import Flask, jsonify, request
from processChat import processRequest
from webCrawl import crawl
from buildEmbeddings import build_embeddings
from customQnA import answer_question, read_embeddings
app = Flask(__name__)

# Sample data (you can replace this with your own data or database connection)
chatRequest = [
    {'id': 1, 'request': 'Description for Task 1'},
    {'id': 2, 'request': 'Description for Task 2'}
]

# Define root domain to crawl
domain = "www.sparkcreativetechnologies.com"
full_url = "https://www.sparkcreativetechnologies.com/"

# crawl(full_url, dir="text.url/", url_dir="urls/" )
# build_embeddings(domain, text_dir="text.url/", url_dir="urls/", process_dir="processed.url/")

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
