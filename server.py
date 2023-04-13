from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import json

class HTTPRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, trans_func, **kwargs):
        self.trans_func = trans_func
        super().__init__(*args, **kwargs)
    
    def do_POST(self):
        content_len = int(self.headers.get('content-length', 0))
        post_body = self.rfile.read(content_len)

        # Convert the received data to a dictionary
        data_dict = json.loads(post_body)

        # Extract the integer and array from the dictionary
        int_val = data_dict['int_val']
        float_arr = np.array(data_dict['float_arr'], dtype='float32')

        # Process the array and integer using the trans_func instance variable
        result = self.trans_func(int_val, float_arr)

        # Convert the result to a string and encode it as JSON
        result_dict = {'result': result}
        result_json = json.dumps(result_dict).encode()

        # Send the result back to the client
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(result_json)


def server(port=8000, address="localhost", trans_func=None):
    httpd = HTTPServer((address, port), lambda *args, **kwargs: HTTPRequestHandler(*args, trans_func=trans_func, **kwargs))
    print('Server started at {}:{}...'.format(address, port))
    httpd.serve_forever()

