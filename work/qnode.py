import json, urllib.request

base = "http://127.0.0.1:8010/api/v1"
tok = json.load(open("/tmp/login.json"))["access_token"]

def get(path):
    req = urllib.request.Request(base + path, headers={"Authorization": "Bearer " + tok})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode()[:4000]
    except Exception as e:
        return getattr(e, "code", "ERR"), str(e)

for p in ["/campaigns/nodes/P295W1R2-N08",
          "/campaigns/nodes?node_id=P295W1R2-N08",
          "/campaigns/nodes?node=P295W1R2-N08",
          "/campaigns?node_id=P295W1R2-N08",
          "/campaigns/plans",
          "/campaigns"]:
    s, b = get(p)
    print("==", p, s)
    print(b[:1500])
    print()
