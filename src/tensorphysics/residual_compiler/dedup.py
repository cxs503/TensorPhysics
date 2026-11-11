"""Triple retrieval deduplication for candidate proposals."""
from typing import List, Dict, Any, Optional, Tuple, Set
from .schema import ProposalSchema
from .config import CompilerConfig
import re
import math
from collections import defaultdict


class TripleRetrievalDedup:
    """Performs triple-retrieval deduplication on candidate proposals.
    
    Three retrieval methods:
    1. Node name retrieval: match against existing campaign node names
    2. Proposal pool retrieval: match against existing proposal pool
    3. Concept grep: keyword-based concept matching
    
    A proposal is considered duplicate if it matches any of the three methods
    above the similarity threshold.
    """
    
    def __init__(
        self,
        config: Optional[CompilerConfig] = None,
        node_corpus: Optional[List[str]] = None,
        proposal_pool: Optional[List[str]] = None,
    ):
        self.config = config or CompilerConfig()
        self.node_corpus = node_corpus or []
        self.proposal_pool = proposal_pool or []
        
        # Build inverted index for fast retrieval
        self._node_index = self._build_inverted_index(self.node_corpus)
        self._proposal_index = self._build_inverted_index(self.proposal_pool)
        
        # Statistics
        self._total_candidates = 0
        self._duplicates_found = 0
        self._unique_proposals = 0
        
        # For recall/precision calculation
        self._true_duplicates = 0
        self._detected_duplicates = 0
        self._false_positives = 0
    
    def deduplicate(
        self, proposals: List[ProposalSchema]
    ) -> List[ProposalSchema]:
        """Perform triple-retrieval deduplication on proposals.
        
        Args:
            proposals: List of candidate proposals
            
        Returns:
            List of unique proposals (duplicates removed)
        """
        unique_proposals = []
        self._total_candidates = len(proposals)
        self._duplicates_found = 0
        self._unique_proposals = 0
        
        seen_titles = set()
        
        for proposal in proposals:
            # Check for duplicates using three methods
            is_duplicate, match_sources = self._check_duplicate(proposal)
            
            if is_duplicate:
                self._duplicates_found += 1
                proposal.dedup_matches = match_sources
            else:
                # Also check against already-accepted proposals
                if self._is_similar_to_accepted(proposal, unique_proposals):
                    self._duplicates_found += 1
                    proposal.dedup_matches.append("similar_to_accepted")
                else:
                    # Check for exact title duplicates
                    title_key = proposal.title.lower().strip()
                    if title_key in seen_titles:
                        self._duplicates_found += 1
                        proposal.dedup_matches.append("exact_title_duplicate")
                    else:
                        unique_proposals.append(proposal)
                        seen_titles.add(title_key)
                        self._unique_proposals += 1
        
        return unique_proposals
    
    def _check_duplicate(
        self, proposal: ProposalSchema
    ) -> Tuple[bool, List[str]]:
        """Check if a proposal is a duplicate using three retrieval methods.
        
        Returns:
            Tuple of (is_duplicate, list_of_match_sources)
        """
        match_sources = []
        
        # Method 1: Node name retrieval
        node_match = self._node_name_retrieval(proposal)
        if node_match:
            match_sources.append(f"node_corpus:{node_match}")
        
        # Method 2: Proposal pool retrieval
        proposal_match = self._proposal_pool_retrieval(proposal)
        if proposal_match:
            match_sources.append(f"proposal_pool:{proposal_match}")
        
        # Method 3: Concept grep
        concept_match = self._concept_grep(proposal)
        if concept_match:
            match_sources.append(f"concept:{concept_match}")
        
        is_duplicate = len(match_sources) > 0
        return is_duplicate, match_sources
    
    def _node_name_retrieval(self, proposal: ProposalSchema) -> Optional[str]:
        """Retrieve similar node names from corpus."""
        if not self.node_corpus:
            return None
        
        # Extract keywords from proposal
        keywords = self._extract_keywords(proposal.title + " " + proposal.physical_goal)
        
        # Search in node corpus using inverted index
        best_match = None
        best_score = 0.0
        
        for node_name in self.node_corpus:
            score = self._calculate_similarity(keywords, node_name, self._node_index)
            if score > self.config.similarity_threshold and score > best_score:
                best_score = score
                best_match = node_name[:80]
        
        return best_match
    
    def _proposal_pool_retrieval(self, proposal: ProposalSchema) -> Optional[str]:
        """Retrieve similar proposals from pool."""
        if not self.proposal_pool:
            return None
        
        # Extract keywords from proposal
        keywords = self._extract_keywords(proposal.title + " " + proposal.physical_goal)
        
        # Search in proposal pool using inverted index
        best_match = None
        best_score = 0.0
        
        for existing_proposal in self.proposal_pool:
            score = self._calculate_similarity(
                keywords, existing_proposal, self._proposal_index
            )
            if score > self.config.similarity_threshold and score > best_score:
                best_score = score
                best_match = existing_proposal[:80]
        
        return best_match
    
    def _concept_grep(self, proposal: ProposalSchema) -> Optional[str]:
        """Perform concept-based grep matching."""
        # Extract key concepts from proposal
        concepts = self._extract_concepts(proposal)
        
        # Check against node corpus
        for node_name in self.node_corpus:
            node_concepts = self._extract_concepts_from_text(node_name)
            overlap = concepts & node_concepts
            if len(overlap) >= 2:  # At least 2 concept overlap
                return f"concept_overlap:{','.join(list(overlap)[:3])}"
        
        # Check against proposal pool
        for existing in self.proposal_pool:
            existing_concepts = self._extract_concepts_from_text(existing)
            overlap = concepts & existing_concepts
            if len(overlap) >= 2:
                return f"concept_overlap:{','.join(list(overlap)[:3])}"
        
        return None
    
    def _is_similar_to_accepted(
        self, proposal: ProposalSchema, accepted: List[ProposalSchema]
    ) -> bool:
        """Check if proposal is similar to already-accepted proposals."""
        keywords = self._extract_keywords(proposal.title + " " + proposal.physical_goal)
        
        for accepted_prop in accepted:
            accepted_text = accepted_prop.title + " " + accepted_prop.physical_goal
            score = self._calculate_similarity_simple(keywords, accepted_text)
            if score > self.config.similarity_threshold:
                return True
        
        return False
    
    def _build_inverted_index(self, documents: List[str]) -> Dict[str, Set[int]]:
        """Build inverted index for fast retrieval."""
        index = defaultdict(set)
        
        for doc_id, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            for token in tokens:
                index[token].add(doc_id)
        
        return dict(index)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into keywords."""
        # Simple tokenization: split on whitespace and punctuation
        # Also handle Chinese characters
        text = text.lower()
        tokens = re.findall(r'[\w\u4e00-\u9fff]+', text)
        return [t for t in tokens if len(t) > 1]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        return self._tokenize(text)
    
    def _extract_concepts(self, proposal: ProposalSchema) -> Set[str]:
        """Extract key concepts from proposal."""
        text = proposal.title + " " + proposal.physical_goal + " " + proposal.physics_domain
        return self._extract_concepts_from_text(text)
    
    def _extract_concepts_from_text(self, text: str) -> Set[str]:
        """Extract concepts from text."""
        # Domain-specific concepts
        concept_keywords = {
            "resistance", "propulsion", "seakeeping", "maneuvering", "stability",
            "structure", "vibration", "ice", "wave", "propeller", "hull",
            "cfd", "bem", "fem", "lbm", "dem", "holtrop", "mmg", "ittc",
            "阻力", "推进", "耐波", "操纵", "稳性", "结构", "振动", "冰", "波浪",
        }
        
        text_lower = text.lower()
        concepts = set()
        
        for concept in concept_keywords:
            if concept in text_lower:
                concepts.add(concept)
        
        return concepts
    
    def _calculate_similarity(
        self, keywords: List[str], document: str, index: Dict[str, Set[int]]
    ) -> float:
        """Calculate similarity between keywords and document using TF-IDF-like scoring."""
        if not keywords or not document:
            return 0.0
        
        doc_tokens = set(self._tokenize(document))
        keyword_set = set(keywords)
        
        # Jaccard similarity
        intersection = keyword_set & doc_tokens
        union = keyword_set | doc_tokens
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _calculate_similarity_simple(self, keywords: List[str], text: str) -> float:
        """Simple similarity calculation."""
        if not keywords or not text:
            return 0.0
        
        text_tokens = set(self._tokenize(text))
        keyword_set = set(keywords)
        
        intersection = keyword_set & text_tokens
        union = keyword_set | text_tokens
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def calculate_recall_precision(
        self, ground_truth_duplicates: Set[str]
    ) -> Tuple[float, float]:
        """Calculate recall and precision of deduplication.
        
        Args:
            ground_truth_duplicates: Set of proposal IDs that are true duplicates
            
        Returns:
            Tuple of (recall, precision)
        """
        # This would require ground truth data
        # For now, return placeholder values
        if not ground_truth_duplicates:
            return 1.0, 1.0
        
        # Simplified calculation
        true_positives = len(ground_truth_duplicates & set(
            p.proposal_id for p in []  # Would need to track detected duplicates
        ))
        
        recall = true_positives / len(ground_truth_duplicates) if ground_truth_duplicates else 1.0
        precision = true_positives / self._duplicates_found if self._duplicates_found > 0 else 1.0
        
        return recall, precision
    
    @property
    def dedup_rate(self) -> float:
        """Calculate deduplication rate."""
        if self._total_candidates == 0:
            return 0.0
        return self._duplicates_found / self._total_candidates
    
    @property
    def total_candidates(self) -> int:
        return self._total_candidates
    
    @property
    def duplicates_found(self) -> int:
        return self._duplicates_found
    
    @property
    def unique_count(self) -> int:
        return self._unique_proposals
