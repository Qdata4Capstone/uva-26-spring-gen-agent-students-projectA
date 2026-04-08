# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Patient Education Agent

A conversational AI agent that helps patients understand medical jargon, conditions, procedures, and medications in plain language. Split into a Node.js/Express backend and a React/Vite frontend.

## Setup

Create `.env` in `server/`:
```
ANTHROPIC_API_KEY=<your_key_here>
NCBI_EMAIL=<your_email_here>
NCBI_API_KEY=<your_key_here>
```

```bash
# Install backend dependencies
cd server && npm install

# Install frontend dependencies
cd client && npm install
```

## Running

```bash
# Terminal 1 — backend (from project root)
cd server && npm run dev
# Server runs at http://localhost:3001

# Terminal 2 — frontend (from project root)
cd client && npm run dev
# Client runs at http://localhost:5173 (or the port Vite prints)
```

## Architecture

```
client/ (React + Vite + Tailwind)
    │  HTTP/WebSocket
server/ (Node.js + Express)
    │  Anthropic SDK — Claude API (chat proxy)
    │  NCBI E-utilities API (medical literature lookup)
```

The backend acts as a proxy between the frontend and the Claude API, keeping the API key server-side. NCBI integration enables the agent to ground answers in PubMed literature when relevant.

## Key Dependencies

- **Backend:** Node.js 18+, Express, Anthropic SDK
- **Frontend:** React, Vite, Tailwind CSS
- **External APIs:** Anthropic Claude (chat), NCBI E-utilities (medical literature)
