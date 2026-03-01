import requests
import os

class APIHelper():

    def __init__(self):
        self.baseUrl = 'http://www.tng-project.org/api/'
        self.headers = {"api-key":"6dcbb9590a97d42c05cd303aa0a59c40"}

    def get(self, path, params=None):
        # make HTTP GET request to path
        r = requests.get(path, params=params, headers=self.headers)

        # raise exception if response code is not HTTP SUCCESS (200)
        r.raise_for_status()

        if r.headers['content-type'] == 'application/json':
            return r.json() # parse json responses automatically
        
        if 'content-disposition' in r.headers:
            filename = r.headers['content-disposition'].split("filename=")[1]
            filepath = os.getcwd() + '/Illustrus/Data/' + filename
            with open(filename, 'wb') as f:
                f.write(r.content)
            return filename # return the filename string
    
        return r
    
    def buildPath(self, order, first, second, third, fourth):
        path = self.baseUrl
        match order:
            case 0: 
                return path 
            case 1: 
                path = path + first
            case 2:
                path = f"{path} {first}/{second}"
            case 3: 
                path = f"{path} {first}/{second}/{third}"
            case 4: 
                path = f"{path} {first}/{second}/{third}/{fourth}"
            case _: 
                print("default base path")
        return path