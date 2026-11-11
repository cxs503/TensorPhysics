import urllib.request, json, os, sys
base='http://127.0.0.1:8010/api/v1'
tok=json.load(open('/tmp/tok.json'))['access_token']
def get(path):
    req=urllib.request.Request(base+path, headers={'Authorization':'Bearer '+tok})
    return json.load(urllib.request.urlopen(req, timeout=60))
if __name__=='__main__':
    print(json.dumps(get(sys.argv[1]), ensure_ascii=False)[:int(sys.argv[2]) if len(sys.argv)>2 else 3000])
