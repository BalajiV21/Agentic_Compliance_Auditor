"""
Tools for the compliance agent
"""
from typing import List, Dict, Optional
from langchain.tools import tool
from loguru import logger
import re
from datetime import datetime


class ComplianceTools:
    """Collection of tools for compliance checking"""

    def __init__(self, retriever):
        """
        Initialize tools with retriever

        Args:
            retriever: Document retriever instance
        """
        self.retriever = retriever

    @staticmethod
    def create_tools(retriever):
        """Create tool instances"""
        tools_instance = ComplianceTools(retriever)

        return [
            tools_instance.search_regulations,
            tools_instance.search_specific_article,
            tools_instance.cross_reference_regulations,
            tools_instance.extract_requirements,
            tools_instance.validate_date,
        ]

    def search_regulations(self, query: str, document_type: Optional[str] = None) -> str:
        """
        Search regulatory documents for relevant information.

        Args:
            query: The compliance question or topic to search for
            document_type: Optional filter (GDPR, HIPAA, SOC2, etc.)

        Returns:
            String containing relevant regulatory excerpts with citations
        """
        logger.info(f"Tool called: search_regulations(query='{query}', document_type='{document_type}')")

        # Build filter if document type specified
        filter_metadata = None
        if document_type:
            filter_metadata = {'document_type': document_type}

        # Retrieve relevant chunks
        results = self.retriever.retrieve(
            query=query,
            top_k=5,
            filter_metadata=filter_metadata
        )

        if not results:
            return f"No relevant information found for query: {query}"

        # Format results
        output = []
        for i, result in enumerate(results, 1):
            citation = result.get('citation', f"[{i}]")
            content = result['content']
            score = result['similarity_score']

            output.append(
                f"{citation}\n"
                f"Relevance: {score:.2%}\n"
                f"Content: {content}\n"
            )

        return "\n---\n".join(output)

    def search_specific_article(self, article_reference: str, regulation: str) -> str:
        """
        Search for a specific article or section in regulations.

        Args:
            article_reference: The article/section number (e.g., "Article 17", "Section 164.502")
            regulation: The regulation name (GDPR, HIPAA, SOC2)

        Returns:
            The content of the specific article
        """
        logger.info(f"Tool called: search_specific_article(article_reference='{article_reference}', regulation='{regulation}')")

        # Search with specific article reference
        query = f"{regulation} {article_reference}"

        results = self.retriever.retrieve(
            query=query,
            top_k=3,
            filter_metadata={'document_type': regulation}
        )

        if not results:
            return f"Could not find {article_reference} in {regulation}"

        # Filter results that actually contain the article reference
        relevant_results = [
            r for r in results
            if article_reference.lower() in r['content'].lower()
        ]

        if not relevant_results:
            return f"Found results for {regulation} but not specifically for {article_reference}"

        # Return the most relevant result
        best_result = relevant_results[0]

        return (
            f"{best_result.get('citation', 'Reference')}\n"
            f"Content: {best_result['content']}"
        )

    def cross_reference_regulations(self, topic: str, regulations: List[str]) -> str:
        """
        Compare how different regulations address the same topic.

        Args:
            topic: The compliance topic to compare (e.g., "data retention", "access control")
            regulations: List of regulations to compare (e.g., ["GDPR", "HIPAA"])

        Returns:
            Comparison of how each regulation addresses the topic
        """
        logger.info(f"Tool called: cross_reference_regulations(topic='{topic}', regulations={regulations})")

        comparisons = []

        for regulation in regulations:
            results = self.retriever.retrieve(
                query=topic,
                top_k=2,
                filter_metadata={'document_type': regulation}
            )

            if results:
                best_result = results[0]
                comparisons.append(
                    f"{regulation}:\n"
                    f"{best_result.get('citation', 'Reference')}\n"
                    f"{best_result['content'][:400]}...\n"
                )
            else:
                comparisons.append(f"{regulation}: No relevant information found\n")

        return "\n---\n".join(comparisons)

    def extract_requirements(self, regulation_text: str) -> str:
        """
        Extract specific requirements or obligations from regulatory text.

        Args:
            regulation_text: Text from a regulation

        Returns:
            List of extracted requirements
        """
        logger.info("Tool called: extract_requirements")

        # Pattern matching for requirements
        patterns = [
            r'(must\s+[^.]+\.)',
            r'(shall\s+[^.]+\.)',
            r'(required\s+to\s+[^.]+\.)',
            r'(is required[^.]+\.)',
            r'(has the right to\s+[^.]+\.)',
        ]

        requirements = []

        for pattern in patterns:
            matches = re.findall(pattern, regulation_text, re.IGNORECASE)
            requirements.extend(matches)

        if not requirements:
            return "No specific requirements identified in the provided text."

        # Remove duplicates and format
        unique_requirements = list(set(requirements))

        output = "Extracted Requirements:\n"
        for i, req in enumerate(unique_requirements, 1):
            output += f"{i}. {req.strip()}\n"

        return output

    def validate_date(self, date_string: str) -> str:
        """
        Validate and parse date strings for compliance checking.

        Args:
            date_string: Date string to validate

        Returns:
            Validation result and parsed date info
        """
        logger.info(f"Tool called: validate_date(date_string='{date_string}')")

        # Common date formats
        formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%B %d, %Y",
            "%d %B %Y",
        ]

        for fmt in formats:
            try:
                parsed_date = datetime.strptime(date_string, fmt)

                # Calculate if date is in the past or future
                now = datetime.now()
                days_diff = (parsed_date - now).days

                result = f"Valid date: {parsed_date.strftime('%B %d, %Y')}\n"

                if days_diff > 0:
                    result += f"This date is {days_diff} days in the future."
                elif days_diff < 0:
                    result += f"This date was {abs(days_diff)} days ago."
                else:
                    result += "This date is today."

                return result

            except ValueError:
                continue

        return f"Could not parse date: {date_string}"


# Convert methods to LangChain tools
def create_langchain_tools(retriever) -> List:
    """
    Create LangChain compatible tools

    Args:
        retriever: Document retriever instance

    Returns:
        List of LangChain tools
    """
    tools_instance = ComplianceTools(retriever)

    @tool
    def search_regulations(query: str, document_type: str = None) -> str:
        """
        Search regulatory documents for relevant information about compliance topics.
        Use this tool when you need to find general information about a compliance topic.

        Args:
            query: The compliance question or topic to search for
            document_type: Optional filter (GDPR, HIPAA, SOC2, etc.)
        """
        return tools_instance.search_regulations(query, document_type)

    @tool
    def search_specific_article(article_reference: str, regulation: str) -> str:
        """
        Search for a specific article or section in regulations.
        Use this when the user asks about a specific article or section number.

        Args:
            article_reference: The article/section number (e.g., "Article 17", "Section 164.502")
            regulation: The regulation name (GDPR, HIPAA, SOC2)
        """
        return tools_instance.search_specific_article(article_reference, regulation)

    @tool
    def cross_reference_regulations(topic: str, regulations: str) -> str:
        """
        Compare how different regulations address the same topic.
        Use this when comparing requirements across multiple regulations.

        Args:
            topic: The compliance topic to compare (e.g., "data retention", "access control")
            regulations: Comma-separated list of regulations (e.g., "GDPR,HIPAA")
        """
        # Parse comma-separated regulations
        regs_list = [r.strip() for r in regulations.split(',')]
        return tools_instance.cross_reference_regulations(topic, regs_list)

    @tool
    def extract_requirements(regulation_text: str) -> str:
        """
        Extract specific requirements or obligations from regulatory text.
        Use this after retrieving regulatory text to identify key requirements.

        Args:
            regulation_text: Text from a regulation
        """
        return tools_instance.extract_requirements(regulation_text)

    return [
        search_regulations,
        search_specific_article,
        cross_reference_regulations,
        extract_requirements,
    ]
