# Presenter Speed Test - Python Edition

This starter uses:

- **Browser JavaScript** to test the presenter's actual connection.
- **Python + FastAPI** to validate the result and keep credentials secret.
- **Google Sheets API** to append a row immediately.
- **Trello REST API** to create a card after the Sheet row is saved.

The Sheet is deliberately saved first. A temporary Trello failure therefore does not discard the test result.

## Project structure

```text
presenter-speedtest-python/
├── main.py                 # FastAPI server, Google Sheets, and Trello
├── static/index.html       # Form and browser speed test
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md
```

## 1. Create the Google Sheet

1. Create a Google Sheet.
2. Add a tab named `Speed Tests`, or use another name and set `GOOGLE_SHEET_NAME`.
3. Copy the spreadsheet ID from its address:

```text
https://docs.google.com/spreadsheets/d/THIS_PART_IS_THE_ID/edit
```

The app writes the header automatically the first time a result is received.

## 2. Create Google credentials

For a small server-to-server app, a service account is the simplest starting point.

1. Create or select a Google Cloud project.
2. Enable the Google Sheets API.
3. Create a service account.
4. Create a JSON key and download it as `service-account.json`.
5. Open the Google Sheet and share it with the service account's email address as an **Editor**.
6. Keep the JSON key private. Never commit it to Git or place it in `static/index.html`.

For a managed production host, prefer the host's secret manager or short-lived Google credentials rather than a long-lived key file.

## 3. Configure the app

Copy the example environment file:

```bash
cp .env.example .env
```

Set at least:

```dotenv
GOOGLE_SPREADSHEET_ID=your_spreadsheet_id
GOOGLE_SHEET_NAME=Speed Tests
GOOGLE_SERVICE_ACCOUNT_FILE=service-account.json
```

The default readiness thresholds are editable in `.env`:

```dotenv
MIN_DOWNLOAD_MBPS=15
MIN_UPLOAD_MBPS=8
MAX_LATENCY_MS=120
MAX_JITTER_MS=30
```

These are workflow defaults, not universal broadcast standards. Adjust them for the production platform and format you use.

## 4. Add Trello, optionally

Set these values in `.env`:

```dotenv
TRELLO_KEY=your_api_key
TRELLO_TOKEN=your_user_token
TRELLO_LIST_ID=the_destination_list_id
```

The Trello key, token, and Google credential remain on the Python server. The browser never receives them.

Without these values, the website still records results in Google Sheets and labels the integration status `Trello not configured`.

## 5. Run locally

Python 3.11 or newer is recommended.

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
fastapi dev main.py
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
fastapi dev main.py
```

Open:

```text
http://127.0.0.1:8000
```

The interactive API documentation is at:

```text
http://127.0.0.1:8000/docs
```

## 6. What happens after the presenter clicks the button

1. The page validates the presenter's name, email, connection type, and consent.
2. Cloudflare's JavaScript module runs download, upload, latency, and jitter measurements in that browser.
3. The page sends the completed result as JSON to `POST /api/submissions`.
4. FastAPI validates the JSON with a Pydantic model.
5. Python appends the row to Google Sheets.
6. Python attempts to create a Trello card.
7. Python writes the Trello card URL or error status back to the same Sheet row.

Do not replace the browser test with a Python `speedtest` package on the web server. That would measure the server's network, not the prospective presenter's network.

## 7. Deploy

The included Dockerfile runs the app with Uvicorn. Any HTTPS-capable container or Python host can run it.

Provide secrets through the hosting platform's environment-variable or secret-management feature. For `GOOGLE_SERVICE_ACCOUNT_JSON`, store the complete JSON object as one environment variable. Do not copy a credential into the Docker image.

Before making the URL broadly public, add at least:

- Rate limiting or a managed web application firewall.
- Bot protection such as a challenge or CAPTCHA.
- A privacy notice and retention policy.
- Restricted access if only invited presenters should submit.
- Monitoring for Google API or Trello failures.

## 8. Use another PM platform

The function to replace is:

```python
create_trello_card(...)
```

You can make a server-side `requests.post(...)` call to Asana, ClickUp, Monday.com, Jira, or another platform. Another simple option is to leave Trello disabled and let Make, Zapier, or Power Automate watch for new Google Sheet rows.

## API payload example

```json
{
  "presenter_name": "Taylor Morgan",
  "email": "taylor@example.com",
  "event_or_role": "Quarterly webinar",
  "location": "Boston, MA",
  "connection_type": "Ethernet",
  "download_mbps": 142.5,
  "upload_mbps": 36.2,
  "latency_ms": 24,
  "jitter_ms": 4,
  "test_duration_seconds": 18.6,
  "timezone": "America/New_York",
  "user_agent": "Browser user agent",
  "effective_network_type": "4g",
  "browser_downlink_mbps": 10,
  "notes": "Using the planned camera and microphone",
  "consent": true,
  "website": ""
}
```
