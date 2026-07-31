# Website

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## Environment Configuration

The website includes built-in API routes under `/api/v1` for the chatbot.

1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Set your OpenAI API key in `website/.env` (or add it to the Vercel project environment variables):
   ```bash
   OPENAI_API_KEY=sk-...
   ```

The chatbot searches the book's `docs/` content automatically. Qdrant is optional: when it is configured, it is used for semantic search; otherwise the assistant falls back to local textbook search. If you want to override the API base URL for a custom deployment, you can set `REACT_APP_API_BASE_URL`, but this is optional.

## Installation

```bash
yarn
```

## Local Development

```bash
yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

Using SSH:

```bash
USE_SSH=true yarn deploy
```

Not using SSH:

```bash
GIT_USER=<Your GitHub username> yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
