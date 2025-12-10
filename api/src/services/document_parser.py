import os
import re
from typing import List, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentParser:
    """
    Service for parsing textbook documents and extracting content for the RAG system.
    """

    @staticmethod
    def parse_markdown_file(file_path: str) -> List[Dict[str, Any]]:
        """
        Parse a markdown file and extract content sections.

        Args:
            file_path: Path to the markdown file

        Returns:
            List of document sections with metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Extract the frontmatter
            frontmatter_match = re.match(r'---\n(.*?)\n---', content, re.DOTALL)
            frontmatter = {}
            if frontmatter_match:
                frontmatter_text = frontmatter_match.group(1)
                for line in frontmatter_text.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        frontmatter[key.strip()] = value.strip().strip('"\'')

            # Remove frontmatter from content
            content_without_frontmatter = re.sub(r'---\n(.*?)\n---', '', content, 1, re.DOTALL).strip()

            # Extract the title from the first H1 header if not in frontmatter
            title_match = re.match(r'#\s+(.+)', content_without_frontmatter)
            title = frontmatter.get('title', title_match.group(1) if title_match else Path(file_path).stem)

            # Split content into sections based on headers
            sections = DocumentParser._split_content_into_sections(content_without_frontmatter, title)

            # Create documents for each section
            documents = []
            for i, section in enumerate(sections):
                doc = {
                    "id": f"{Path(file_path).stem}_{i}",
                    "content": section['content'],
                    "chapter": frontmatter.get('title', title),
                    "section": section['header'],
                    "title": f"{title} - {section['header']}" if section['header'] else title,
                    "url": f"/docs/{Path(file_path).stem}"  # Assuming docs are served from /docs/
                }
                documents.append(doc)

            logger.info(f"Parsed {len(documents)} sections from {file_path}")
            return documents

        except Exception as e:
            logger.error(f"Error parsing markdown file {file_path}: {e}")
            raise

    @staticmethod
    def _split_content_into_sections(content: str, chapter_title: str) -> List[Dict[str, str]]:
        """
        Split markdown content into sections based on headers.

        Args:
            content: The markdown content
            chapter_title: The title of the chapter

        Returns:
            List of sections with headers and content
        """
        # Split content by H2 headers (##)
        h2_pattern = r'\n##\s+(.+?)(?=\n##\s+|\n# ##\s+|\Z)'
        sections = []

        # First, get content before any H2 headers
        first_match = re.search(h2_pattern, content)
        if first_match:
            before_content = content[:first_match.start()].strip()
            if before_content:
                sections.append({
                    "header": "Introduction",
                    "content": before_content
                })

            # Process each H2 section
            h2_parts = re.split(h2_pattern, content)
            i = 1  # Start from the first header name
            while i < len(h2_parts):
                header = h2_parts[i].strip()
                if i + 1 < len(h2_parts):
                    section_content = h2_parts[i + 1]
                    # Get content up to the next H2 header
                    next_h2_match = re.search(h2_pattern, section_content)
                    if next_h2_match:
                        section_content = section_content[:next_h2_match.start()].strip()
                    else:
                        section_content = section_content.strip()

                    if section_content:
                        sections.append({
                            "header": header,
                            "content": section_content
                        })
                i += 2
        else:
            # If no H2 headers, treat entire content as one section
            sections.append({
                "header": chapter_title,
                "content": content
            })

        return sections

    @staticmethod
    def parse_all_textbook_docs(docs_dir: str) -> List[Dict[str, Any]]:
        """
        Parse all markdown files in the textbook docs directory.

        Args:
            docs_dir: Path to the docs directory

        Returns:
            List of all document sections from all files
        """
        all_documents = []

        for root, dirs, files in os.walk(docs_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    try:
                        documents = DocumentParser.parse_markdown_file(file_path)
                        all_documents.extend(documents)
                    except Exception as e:
                        logger.error(f"Failed to parse {file_path}: {e}")

        logger.info(f"Parsed a total of {len(all_documents)} sections from textbook")
        return all_documents

# Global instance
document_parser = DocumentParser()