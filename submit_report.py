"""Submit the report via API."""
import json
import urllib.request
import urllib.error

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJlY29fYWRtaW4iLCJ1c2VyX2lkIjoxLCJyb2xlIjoiQURNSU4iLCJleHAiOjE3OTM0MzcwMzAsInR5cGUiOiJhY2Nlc3MiLCJpYXQiOjE3OTM0MzUyMzB9.tkPUvQq0eItEaly1r2tJenpmRCmU08FnC1C2M8vziNU"

with open('/tmp/report.json', 'r') as f:
    report_data = f.read()

url = "http://127.0.0.1:8010/api/v1/tensor/reports"
req = urllib.request.Request(
    url,
    data=report_data.encode('utf-8'),
    headers={
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    },
    method="POST"
)

try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = resp.read().decode('utf-8')
        print(f"Status: {resp.status}")
        print(f"Response: {result[:500]}")
except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code}")
    print(f"Response: {e.read().decode('utf-8')[:500]}")
except Exception as e:
    print(f"Error: {e}")
