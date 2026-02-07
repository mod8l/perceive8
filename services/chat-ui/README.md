# perceive8 Chat UI

Chainlit-based chat interface for the perceive8 audio analysis API.

## Setup

1. **Install dependencies:**

   ```bash
   poetry install
   ```

2. **Configure environment:**

   ```bash
   cp .env.example .env
   # Edit .env with your perceive8 API URL if needed
   ```

3. **Run the app:**

   ```bash
   poetry run chainlit run app.py
   ```

   The UI will be available at `http://localhost:8000` (Chainlit default port).

   > If the perceive8 API is also on port 8000, run Chainlit on a different port:
   > ```bash
   > chainlit run app.py --port 8080
   > ```

## Docker

```bash
docker build -t perceive8-chat-ui .
docker run -p 8080:8080 --env-file .env perceive8-chat-ui
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PERCEIVE8_API_URL` | `http://localhost:8000` | Base URL of the perceive8 API |
| `DEFAULT_USER_ID` | `default-user` | User ID sent with analysis requests |
