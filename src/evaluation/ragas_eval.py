"""
RAGAS evaluation framework for compliance RAG system
Evaluates faithfulness, answer relevancy, context precision, and context recall
"""
from typing import List, Dict
from loguru import logger
import json
from pathlib import Path

# Note: RAGAS requires specific versions and setup
# For now, we'll create the framework structure
# Full RAGAS integration requires: ragas, datasets, and proper LLM setup


class ComplianceEvaluator:
    """
    Evaluator for compliance RAG system
    """

    def __init__(self, retriever, agent):
        """
        Initialize evaluator

        Args:
            retriever: Document retriever
            agent: Compliance agent
        """
        self.retriever = retriever
        self.agent = agent

    def create_test_dataset(self) -> List[Dict]:
        """
        Create a test dataset for evaluation

        Returns:
            List of test cases with questions and ground truth
        """
        test_cases = [
            {
                "question": "What is the right to erasure under GDPR?",
                "ground_truth": "The right to erasure, also known as the right to be forgotten, is defined in GDPR Article 17. It gives data subjects the right to obtain from the controller the erasure of personal data concerning them without undue delay.",
                "context_filter": {"document_type": "GDPR"}
            },
            {
                "question": "What are the requirements for data retention?",
                "ground_truth": "Data should be kept in a form which permits identification of data subjects for no longer than is necessary for the purposes for which the personal data are processed (storage limitation principle).",
                "context_filter": {"document_type": "GDPR"}
            },
            {
                "question": "What is Protected Health Information under HIPAA?",
                "ground_truth": "Protected Health Information (PHI) is information that relates to the past, present, or future physical or mental health or condition of an individual, the provision of health care to an individual, or payment for health care.",
                "context_filter": {"document_type": "HIPAA"}
            },
            {
                "question": "What are the administrative safeguards required by HIPAA?",
                "ground_truth": "HIPAA Section 164.308 requires administrative safeguards including security management process, assigned security responsibility, workforce security, information access management, and security awareness training.",
                "context_filter": {"document_type": "HIPAA"}
            },
            {
                "question": "What are the five Trust Services Criteria in SOC2?",
                "ground_truth": "The five Trust Services Criteria are: Security, Availability, Processing Integrity, Confidentiality, and Privacy.",
                "context_filter": {"document_type": "SOC2"}
            }
        ]

        return test_cases

    def evaluate_retrieval(
        self,
        query: str,
        ground_truth_context: str,
        filter_metadata: Dict = None
    ) -> Dict:
        """
        Evaluate retrieval quality

        Args:
            query: Query string
            ground_truth_context: Expected context
            filter_metadata: Metadata filters

        Returns:
            Evaluation metrics
        """
        # Retrieve documents
        results = self.retriever.retrieve(
            query=query,
            top_k=5,
            filter_metadata=filter_metadata
        )

        # Calculate metrics
        metrics = {
            "num_results": len(results),
            "avg_similarity": sum(r['similarity_score'] for r in results) / len(results) if results else 0,
            "top_similarity": results[0]['similarity_score'] if results else 0
        }

        # Check if ground truth is in retrieved contexts
        retrieved_text = " ".join([r['content'] for r in results])
        metrics["contains_ground_truth"] = ground_truth_context.lower() in retrieved_text.lower()

        return metrics

    def evaluate_answer_quality(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        sources: List[Dict]
    ) -> Dict:
        """
        Evaluate answer quality (simplified version)

        Args:
            question: Question asked
            answer: Generated answer
            ground_truth: Expected answer
            sources: Retrieved sources

        Returns:
            Quality metrics
        """
        metrics = {}

        # 1. Length metrics
        metrics["answer_length"] = len(answer)
        metrics["has_sufficient_length"] = len(answer) > 50

        # 2. Citation metrics
        citation_indicators = ["[", "Article", "Section", "Source:", "according to"]
        metrics["has_citations"] = any(indicator in answer for indicator in citation_indicators)

        # 3. Source usage
        metrics["num_sources"] = len(sources)
        metrics["uses_sources"] = len(sources) > 0

        # 4. Keyword overlap with ground truth
        answer_words = set(answer.lower().split())
        truth_words = set(ground_truth.lower().split())
        overlap = len(answer_words & truth_words)
        metrics["keyword_overlap"] = overlap / len(truth_words) if truth_words else 0

        # 5. Specificity check (contains numbers, specific terms)
        metrics["contains_specifics"] = any(char.isdigit() for char in answer)

        return metrics

    def run_evaluation(self, test_cases: List[Dict] = None) -> Dict:
        """
        Run full evaluation on test cases

        Args:
            test_cases: Optional test cases (uses default if None)

        Returns:
            Evaluation results
        """
        if test_cases is None:
            test_cases = self.create_test_dataset()

        logger.info(f"Running evaluation on {len(test_cases)} test cases")

        results = []

        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"Evaluating test case {i}/{len(test_cases)}")

            question = test_case["question"]
            ground_truth = test_case["ground_truth"]
            filter_metadata = test_case.get("context_filter")

            # Run agent
            agent_result = self.agent.run(question)

            # Evaluate retrieval
            retrieval_metrics = self.evaluate_retrieval(
                query=question,
                ground_truth_context=ground_truth,
                filter_metadata=filter_metadata
            )

            # Evaluate answer
            answer_metrics = self.evaluate_answer_quality(
                question=question,
                answer=agent_result['answer'],
                ground_truth=ground_truth,
                sources=agent_result['sources']
            )

            result = {
                "question": question,
                "answer": agent_result['answer'],
                "ground_truth": ground_truth,
                "retrieval_metrics": retrieval_metrics,
                "answer_metrics": answer_metrics,
                "iterations": agent_result.get('iterations', 0)
            }

            results.append(result)

        # Calculate aggregate metrics
        aggregate = self._calculate_aggregate_metrics(results)

        evaluation_report = {
            "test_cases": len(test_cases),
            "results": results,
            "aggregate_metrics": aggregate
        }

        logger.info("Evaluation complete")

        return evaluation_report

    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics across all results"""
        if not results:
            return {}

        aggregate = {
            "avg_similarity": sum(r['retrieval_metrics']['avg_similarity'] for r in results) / len(results),
            "avg_keyword_overlap": sum(r['answer_metrics']['keyword_overlap'] for r in results) / len(results),
            "pct_with_citations": sum(1 for r in results if r['answer_metrics']['has_citations']) / len(results) * 100,
            "pct_with_specifics": sum(1 for r in results if r['answer_metrics']['contains_specifics']) / len(results) * 100,
            "avg_iterations": sum(r['iterations'] for r in results) / len(results)
        }

        return aggregate

    def save_evaluation_report(self, report: Dict, output_path: str):
        """Save evaluation report to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Evaluation report saved to {output_path}")

    def print_evaluation_summary(self, report: Dict):
        """Print a summary of evaluation results"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)

        agg = report['aggregate_metrics']

        print(f"\n📊 Aggregate Metrics:")
        print(f"  - Average Similarity Score: {agg['avg_similarity']:.2%}")
        print(f"  - Average Keyword Overlap: {agg['avg_keyword_overlap']:.2%}")
        print(f"  - Answers with Citations: {agg['pct_with_citations']:.1f}%")
        print(f"  - Answers with Specifics: {agg['pct_with_specifics']:.1f}%")
        print(f"  - Average Iterations: {agg['avg_iterations']:.1f}")

        print(f"\n📝 Individual Results:")
        for i, result in enumerate(report['results'], 1):
            print(f"\n  Test Case {i}: {result['question'][:60]}...")
            print(f"    ✓ Top Similarity: {result['retrieval_metrics']['top_similarity']:.2%}")
            print(f"    ✓ Has Citations: {'Yes' if result['answer_metrics']['has_citations'] else 'No'}")
            print(f"    ✓ Keyword Overlap: {result['answer_metrics']['keyword_overlap']:.2%}")

        print("\n" + "="*60)


# Benchmark dataset for compliance questions
BENCHMARK_DATASET = [
    {
        "question": "What personal data rights does GDPR Article 15 provide?",
        "ground_truth": "Article 15 provides the right of access, allowing data subjects to obtain confirmation of whether their personal data is being processed and access to that data.",
        "regulation": "GDPR"
    },
    {
        "question": "What is the deadline for breach notification under HIPAA?",
        "ground_truth": "Covered entities must notify affected individuals within 60 days of discovery of a breach.",
        "regulation": "HIPAA"
    },
    {
        "question": "What does data minimization mean in GDPR?",
        "ground_truth": "Data minimization means that personal data shall be adequate, relevant and limited to what is necessary in relation to the purposes for which they are processed.",
        "regulation": "GDPR"
    }
]


if __name__ == "__main__":
    # Test the evaluator
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).parent.parent))

    from retrieval import VectorStore, CitationRetriever
    from agents import ComplianceAgent, create_langchain_tools

    # Initialize components
    vector_store = VectorStore(
        persist_directory="./data/chroma_db",
        collection_name="test_compliance"
    )
    retriever = CitationRetriever(vector_store)
    tools = create_langchain_tools(retriever)

    agent = ComplianceAgent(
        retriever=retriever,
        tools=tools,
        model_name="llama3.2",
        max_iterations=2  # Reduced for faster evaluation
    )

    # Create evaluator
    evaluator = ComplianceEvaluator(retriever, agent)

    # Run evaluation
    print("Starting evaluation...")
    report = evaluator.run_evaluation()

    # Print summary
    evaluator.print_evaluation_summary(report)

    # Save report
    output_path = "./evaluation_report.json"
    evaluator.save_evaluation_report(report, output_path)
    print(f"\nFull report saved to: {output_path}")
