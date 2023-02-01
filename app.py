from flask import Flask, Response, jsonify, request
from dotenv import load_dotenv
from langchain_bot import print_answer, source_index

load_dotenv()

app = Flask(__name__)


@app.route('/health/', methods=['GET'])
def healthCheck():
    return Response('{"message":"I am here..."}', status=200, mimetype='application/json')


@app.route('/build-index/', methods=['GET'])
def buildIndex():
    source_index()
    return jsonify({
        "answer": "Building Index..."
    }), 200


@app.route('/ask/', methods=['POST'])
def askQuestion():
    app.logger.info('Answer requested...')

    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        query = json["query"]
    else:
        return jsonify({
            "error": "Content-Type not supported!"
        }), 501

    if not query:
        return jsonify({
            "error": "Invalid query"
        }), 501

    return jsonify({
        "answer": print_answer(query)
    }), 200


if __name__ == '__main__':
    app.run(hots='0.0.0.0', port=5000)
