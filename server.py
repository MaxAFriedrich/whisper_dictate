import main
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import json

class HTTPRequestHandler(BaseHTTPRequestHandler):
    
    def do_POST(self):
        content_len = int(self.headers.get('content-length', 0))
        post_body = self.rfile.read(content_len)

        # Convert the received data to a dictionary
        data_dict = json.loads(post_body)

        # Extract the integer and array from the dictionary
        int_val = data_dict['int_val']
        float_arr = np.array(data_dict['float_arr'], dtype='float32')

        # Process the array and integer
        result = main.transcribe(int_val, float_arr)

        # Convert the result to a string and encode it as JSON
        result_dict = {'result': result}
        result_json = json.dumps(result_dict).encode()

        # Send the result back to the client
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(result_json)


def server(port=8000, address="localhost"):
    httpd = HTTPServer((address, port), HTTPRequestHandler)
    print('Server started at localhost:{}...'.format(port))
    httpd.serve_forever()

if __name__ == "__main__":
    server()
