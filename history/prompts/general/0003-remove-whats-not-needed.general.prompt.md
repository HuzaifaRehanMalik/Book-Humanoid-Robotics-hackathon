---
id: 0003
title: remove whats not needed
stage: general
date: 2026-06-30
surface: agent
model: Raptor mini
feature: none
branch: unknown
user: unknown
command: remove legacy backend and docs
labels: [cleanup,docs,api]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - README.md
 - .gitignore
 - website/README.md
 - website/.env.example
 - website/src/components/SidebarChat/SidebarChat.tsx
tests:
 - none
---

## Prompt

remove whats not needed

## Response snapshot

Removed the legacy root Python API directory and Procfile, updated root README and .gitignore to reflect the website-hosted TypeScript API, and updated website docs and chat component routing so the app uses `/api/v1` from `website/api/v1`.

## Outcome

- ✅ Impact: cleaned up legacy backend artifacts and aligned docs with website-hosted API
- 🧪 Tests: none
- 📁 Files: `README.md`, `.gitignore`, `website/README.md`, `website/.env.example`, `website/src/components/SidebarChat/SidebarChat.tsx`
- 🔁 Next prompts: verify website API functions compile and run in Vercel environment
- 🧠 Reflection: removed Python backend remnants while preserving the new TS API implementation

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): none
- Prompt variant (if applicable): none
- Next experiment (smallest change to try): run `npm --prefix website run typecheck` to validate TS API and frontend
