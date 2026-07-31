// Vercel discovers functions from the repository-root `api/` directory.
// Keep the chatbot implementation in the Docusaurus app and expose it here.
export {default} from '../../website/api/v1/chat';
