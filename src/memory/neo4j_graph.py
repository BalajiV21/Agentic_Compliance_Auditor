"""
Neo4j-based knowledge graph for compliance relationships
Stores relationships between regulations, articles, requirements, and entities
"""
from neo4j import GraphDatabase
from typing import List, Dict, Optional
from loguru import logger


class ComplianceKnowledgeGraph:
    """
    Knowledge graph for compliance relationships using Neo4j
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password"
    ):
        """
        Initialize Neo4j connection

        Args:
            uri: Neo4j connection URI
            user: Username
            password: Password
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {uri}")
            self._create_constraints()
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j: {e}. Using fallback mode.")
            self.driver = None
            self.graph_fallback = {
                'nodes': {},
                'relationships': []
            }

    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()

    def _create_constraints(self):
        """Create uniqueness constraints for node types"""
        if not self.driver:
            return

        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Regulation) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (req:Requirement) REQUIRE req.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        ]

        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.debug(f"Constraint already exists or error: {e}")

    def add_regulation(
        self,
        regulation_id: str,
        name: str,
        description: Optional[str] = None,
        properties: Optional[Dict] = None
    ):
        """
        Add a regulation node

        Args:
            regulation_id: Unique identifier (e.g., "GDPR")
            name: Regulation name
            description: Optional description
            properties: Additional properties
        """
        props = properties or {}
        props.update({
            'id': regulation_id,
            'name': name,
            'description': description or ''
        })

        if self.driver:
            with self.driver.session() as session:
                session.run(
                    """
                    MERGE (r:Regulation {id: $id})
                    SET r += $props
                    """,
                    id=regulation_id,
                    props=props
                )
            logger.debug(f"Added regulation: {name}")
        else:
            self.graph_fallback['nodes'][regulation_id] = {
                'type': 'Regulation',
                **props
            }

    def add_article(
        self,
        article_id: str,
        regulation_id: str,
        title: str,
        content: Optional[str] = None
    ):
        """
        Add an article/section node

        Args:
            article_id: Unique identifier (e.g., "GDPR_Article_17")
            regulation_id: Parent regulation ID
            title: Article title
            content: Article content
        """
        if self.driver:
            with self.driver.session() as session:
                session.run(
                    """
                    MERGE (a:Article {id: $article_id})
                    SET a.title = $title, a.content = $content
                    WITH a
                    MATCH (r:Regulation {id: $regulation_id})
                    MERGE (r)-[:CONTAINS]->(a)
                    """,
                    article_id=article_id,
                    regulation_id=regulation_id,
                    title=title,
                    content=content or ''
                )
            logger.debug(f"Added article: {title}")
        else:
            self.graph_fallback['nodes'][article_id] = {
                'type': 'Article',
                'id': article_id,
                'title': title,
                'content': content
            }
            self.graph_fallback['relationships'].append({
                'from': regulation_id,
                'to': article_id,
                'type': 'CONTAINS'
            })

    def add_requirement(
        self,
        requirement_id: str,
        article_id: str,
        description: str,
        requirement_type: str = "MUST"
    ):
        """
        Add a requirement node

        Args:
            requirement_id: Unique identifier
            article_id: Parent article ID
            description: Requirement description
            requirement_type: Type (MUST, SHOULD, MAY)
        """
        if self.driver:
            with self.driver.session() as session:
                session.run(
                    """
                    MERGE (req:Requirement {id: $requirement_id})
                    SET req.description = $description, req.type = $type
                    WITH req
                    MATCH (a:Article {id: $article_id})
                    MERGE (a)-[:SPECIFIES]->(req)
                    """,
                    requirement_id=requirement_id,
                    article_id=article_id,
                    description=description,
                    type=requirement_type
                )
            logger.debug(f"Added requirement: {requirement_id}")
        else:
            self.graph_fallback['nodes'][requirement_id] = {
                'type': 'Requirement',
                'id': requirement_id,
                'description': description,
                'requirement_type': requirement_type
            }
            self.graph_fallback['relationships'].append({
                'from': article_id,
                'to': requirement_id,
                'type': 'SPECIFIES'
            })

    def add_cross_reference(
        self,
        from_article_id: str,
        to_article_id: str,
        relationship_type: str = "REFERENCES"
    ):
        """
        Add a cross-reference between articles

        Args:
            from_article_id: Source article
            to_article_id: Target article
            relationship_type: Type of relationship
        """
        if self.driver:
            with self.driver.session() as session:
                session.run(
                    f"""
                    MATCH (a1:Article {{id: $from_id}})
                    MATCH (a2:Article {{id: $to_id}})
                    MERGE (a1)-[:{relationship_type}]->(a2)
                    """,
                    from_id=from_article_id,
                    to_id=to_article_id
                )
            logger.debug(f"Added cross-reference: {from_article_id} -> {to_article_id}")
        else:
            self.graph_fallback['relationships'].append({
                'from': from_article_id,
                'to': to_article_id,
                'type': relationship_type
            })

    def find_related_articles(
        self,
        article_id: str,
        max_depth: int = 2
    ) -> List[Dict]:
        """
        Find related articles through relationships

        Args:
            article_id: Starting article ID
            max_depth: Maximum relationship depth

        Returns:
            List of related articles
        """
        if self.driver:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH path = (a:Article {id: $article_id})-[*1..%d]-(related:Article)
                    RETURN DISTINCT related.id as id, related.title as title, length(path) as distance
                    ORDER BY distance
                    """ % max_depth,
                    article_id=article_id
                )

                return [dict(record) for record in result]
        else:
            # Simple fallback implementation
            related = []
            for rel in self.graph_fallback['relationships']:
                if rel['from'] == article_id:
                    node = self.graph_fallback['nodes'].get(rel['to'])
                    if node and node['type'] == 'Article':
                        related.append(node)

            return related

    def get_regulation_overview(self, regulation_id: str) -> Dict:
        """
        Get an overview of a regulation's structure

        Args:
            regulation_id: Regulation ID

        Returns:
            Overview dictionary with counts and structure
        """
        if self.driver:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (r:Regulation {id: $regulation_id})
                    OPTIONAL MATCH (r)-[:CONTAINS]->(a:Article)
                    OPTIONAL MATCH (a)-[:SPECIFIES]->(req:Requirement)
                    RETURN r.name as name,
                           count(DISTINCT a) as article_count,
                           count(DISTINCT req) as requirement_count
                    """,
                    regulation_id=regulation_id
                )

                record = result.single()
                if record:
                    return dict(record)

        return {
            'name': regulation_id,
            'article_count': 0,
            'requirement_count': 0
        }

    def search_by_topic(self, topic: str) -> List[Dict]:
        """
        Search articles and requirements by topic

        Args:
            topic: Search term

        Returns:
            Matching nodes
        """
        if self.driver:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (n)
                    WHERE (n:Article OR n:Requirement)
                      AND (toLower(n.title) CONTAINS toLower($topic)
                           OR toLower(n.description) CONTAINS toLower($topic)
                           OR toLower(n.content) CONTAINS toLower($topic))
                    RETURN n.id as id, labels(n)[0] as type, coalesce(n.title, n.description) as text
                    LIMIT 10
                    """,
                    topic=topic
                )

                return [dict(record) for record in result]
        else:
            # Fallback search
            results = []
            for node_id, node in self.graph_fallback['nodes'].items():
                text = str(node).lower()
                if topic.lower() in text:
                    results.append({
                        'id': node_id,
                        'type': node['type'],
                        'text': node.get('title') or node.get('description', '')
                    })

            return results[:10]

    def clear_graph(self):
        """Clear all nodes and relationships (use with caution!)"""
        if self.driver:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Cleared entire graph")
        else:
            self.graph_fallback = {
                'nodes': {},
                'relationships': []
            }


def populate_sample_graph(kg: ComplianceKnowledgeGraph):
    """
    Populate graph with sample data from our regulatory documents

    Args:
        kg: Knowledge graph instance
    """
    logger.info("Populating sample knowledge graph...")

    # Add regulations
    kg.add_regulation("GDPR", "General Data Protection Regulation", "EU data protection law")
    kg.add_regulation("HIPAA", "Health Insurance Portability and Accountability Act", "US healthcare privacy law")
    kg.add_regulation("SOC2", "Service Organization Control 2", "Trust services criteria")

    # Add GDPR articles
    kg.add_article(
        "GDPR_Article_5",
        "GDPR",
        "Article 5 - Principles relating to processing",
        "Personal data shall be processed lawfully, fairly and transparently..."
    )
    kg.add_article(
        "GDPR_Article_17",
        "GDPR",
        "Article 17 - Right to erasure (right to be forgotten)",
        "The data subject shall have the right to obtain erasure of personal data..."
    )

    # Add requirements
    kg.add_requirement(
        "GDPR_Article_17_Req_1",
        "GDPR_Article_17",
        "Controller must erase personal data without undue delay when no longer necessary",
        "MUST"
    )

    # Add HIPAA sections
    kg.add_article(
        "HIPAA_164_502",
        "HIPAA",
        "Section 164.502 - Uses and disclosures of PHI",
        "A covered entity may not use or disclose protected health information..."
    )

    # Add cross-reference
    kg.add_cross_reference("GDPR_Article_17", "GDPR_Article_5", "RELATES_TO")

    logger.info("Sample graph populated successfully")


if __name__ == "__main__":
    # Test the knowledge graph
    kg = ComplianceKnowledgeGraph()

    # Populate with sample data
    populate_sample_graph(kg)

    # Test queries
    print("\n=== GDPR Overview ===")
    overview = kg.get_regulation_overview("GDPR")
    print(overview)

    print("\n=== Search for 'erasure' ===")
    results = kg.search_by_topic("erasure")
    for result in results:
        print(f"{result['type']}: {result['text']}")

    print("\n=== Related Articles ===")
    related = kg.find_related_articles("GDPR_Article_17")
    for article in related:
        print(f"- {article.get('title', article.get('id'))}")

    # Clean up
    kg.close()
