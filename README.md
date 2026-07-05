# Physical AI & Humanoid Robotics Textbook

An AI-native textbook on Physical AI and Humanoid Robotics, built with Docusaurus, Claude Code, and Spec-Kit Plus.

## Overview

This textbook provides comprehensive coverage of Physical AI and Humanoid Robotics, including:

- Foundations of robotics and AI integration
- Control systems and locomotion
- Sensor and actuator technologies
- Learning algorithms and adaptation
- Ethics and safety considerations
- Future directions and applications

## Features

- Complete textbook content on Physical AI & Humanoid Robotics
- Interactive documentation with Docusaurus
- AI-native content creation and maintenance
- Integrated RAG chatbot for Q&A with textbook content
- Mobile-responsive design
- Search functionality

## Structure

The textbook is organized into several key sections:

1. **Foundations**: Basic concepts of Physical AI and humanoid robotics
2. **Core Systems**: Control, sensing, and actuation systems
3. **Advanced Topics**: Learning, ethics, and future directions

## Development

This project uses:
- Docusaurus for documentation
- TypeScript for type safety
- React for interactive components
- Claude Code for AI-assisted development

## Running Locally

### Frontend (Website)
1. Navigate to the `website` directory
2. Install dependencies: `npm install`
3. Start the development server: `npm run start`

### Chat API
The chatbot API is implemented inside `website/api/v1` as TypeScript serverless functions.

- No separate backend server is required.
- In local development, run the website and the API together through Docusaurus/Vercel tooling.
- Set `OPENAI_API_KEY` in `website/.env` or in your Vercel environment.
- The frontend calls `/api/v1/chat` and `/api/v1/chat-selected-text` directly.

### Environment Configuration
- Set `OPENAI_API_KEY` in `website/.env` or in Vercel environment variables.
- If using a custom API base URL, set `REACT_APP_API_BASE_URL` in the website environment.

## Deployment

### Website Deployment
This project deploys the Docusaurus website and chatbot API together on Vercel.

- The website and API functions live under `website/`.
- Vercel routes `/api/*` to the TypeScript handlers in `website/api/`.
- No separate Python backend deployment is required.

### Frontend Deployment
The Docusaurus frontend can be deployed to platforms like Vercel, Netlify, or GitHub Pages following standard Docusaurus deployment practices.

## Contributing

This project uses Spec-Kit Plus and Claude Code for structured development. See the `specs/` directory for feature specifications and implementation plans.

## License

This textbook is open source and available under the MIT License.
