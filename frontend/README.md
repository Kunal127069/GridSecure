# GridSecure Frontend

Static frontend dashboard for the GridSecure electricity theft detection project.

## Files
- `index.html` — application shell and pages
- `css/styles.css` — complete UI styling
- `js/app.js` — navigation, frontend interactions and API integration points
- `assets/` — place images/icons here if needed

## Pages
1. Overview
2. Consumer Investigation
3. Risk Assessment
4. Model & Analytics

## Backend
The frontend is designed to connect to the GridSecure FastAPI backend at `http://127.0.0.1:8000`. The project API exposes `/health`, `/analytics`, `/metrics`, `/consumer/{consumer_id}` and `/predict`.

Start the backend from the GridSecure project root with:

```bash
uvicorn src.api:app --reload
```

If the browser blocks API requests when opening `index.html` directly, serve the frontend folder with:

```bash
python -m http.server 5500
```

Then open `http://127.0.0.1:5500`.

## GitHub
Put this folder inside the repository, for example:

```text
GridSecure/
├── data/
├── docs/
├── models/
├── src/
└── frontend/
    ├── index.html
    ├── css/styles.css
    ├── js/app.js
    ├── assets/
    ├── README.md
    └── .gitignore
```

Then run:

```bash
git add frontend
git commit -m "Add GridSecure frontend"
git push
```

The frontend does not contain the ML model; predictions remain on the Python/FastAPI backend.
