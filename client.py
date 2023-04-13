import json
import urllib.request

def client(int_val, float_arr, host:str = "localhost",port:int = 8000):
    url = f"http://{host}:{str(port)}"
    # Convert the integer and array to a dictionary
    data_dict = {'int_val': int_val, 'float_arr': float_arr.tolist()}

    # Encode the data as JSON
    data_json = json.dumps(data_dict).encode()

    # Set the headers and make the request
    headers = {'Content-type': 'application/json'}
    req = urllib.request.Request(url, data=data_json, headers=headers)
    response = urllib.request.urlopen(req)

    # Read the response and decode it from JSON
    response_json = response.read().decode()
    response_dict = json.loads(response_json)

    # Extract the result from the dictionary
    result = response_dict['result']

    return result
