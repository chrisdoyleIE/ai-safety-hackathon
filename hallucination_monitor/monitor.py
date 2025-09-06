import os
import asyncio
import math
from typing import List, Optional
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from enum import Enum


class EntailmentRelation(str, Enum):
    """Possible entailment relations between two texts"""
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


class EntailmentResponse(BaseModel):
    """Structured response for entailment checking"""
    relation: EntailmentRelation
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    reasoning: Optional[str] = Field(None, description="Brief explanation of the entailment decision")


class SemanticCluster(BaseModel):
    """Represents a cluster of semantically equivalent responses"""
    responses: List[str] = Field(description="All responses in this cluster")
    representative: Optional[str] = Field(None, description="Representative response for this cluster")
    size: Optional[int] = Field(None, ge=1, description="Number of responses in cluster")
    
    def model_post_init(self, __context):
        """Set defaults after validation"""
        if not self.representative and self.responses:
            # Use first response as representative if not specified
            self.representative = self.responses[0]
        if self.size is None:
            self.size = len(self.responses)


class ConfabulationResult(BaseModel):
    """Complete result of confabulation detection analysis"""
    response: str = Field(description="Best response generated at low temperature")
    semantic_entropy: float = Field(ge=0.0, description="Calculated semantic entropy")
    likely_confabulation: bool = Field(description="Whether this is likely a confabulation")
    num_clusters: int = Field(ge=0, description="Number of semantic clusters found")
    num_responses: int = Field(ge=0, description="Total number of responses generated")
    clusters: List[SemanticCluster] = Field(description="Detailed cluster information")
    all_responses: List[str] = Field(description="All generated responses")


class SemanticEntropyLLM:
    """
    Semantic entropy implementation for detecting confabulations in LLM outputs.
    Based on "Detecting hallucinations in large language models using semantic entropy"
    by Farquhar et al. (Nature 2024)
    
    Core concept: Confabulations (arbitrary/incorrect generations) have high semantic entropy
    because the model generates different meanings across multiple samples, while factual
    responses tend to be semantically consistent even if token-level variation exists.
    
    Uses OpenAI with structured outputs and Pydantic models.
    """
    
    def __init__(
        self, 
        generation_model: str = "gpt-4o-2024-08-06",
        entailment_model: str = "gpt-4o-2024-08-06", 
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the SemanticEntropyLLM
        
        Args:
            generation_model: OpenAI model for response generation
            entailment_model: OpenAI model for entailment checking
            openai_api_key: OpenAI API key (or from OPENAI_API_KEY env var)
        """
        if openai_api_key:
            self.client = AsyncOpenAI(api_key=openai_api_key)
        else:
            self.client = AsyncOpenAI()
            
        self.generation_model = generation_model
        self.entailment_model = entailment_model
    
    async def generate_responses(self, prompt: str, n: int = 1, temperature: float = 1.0) -> List[str]:
        """
        Generate n responses using OpenAI's native batch generation.
        
        Paper context (Section 4.1): To compute semantic entropy, we need multiple samples
        from p(y|x) where y is the output and x is the input. Since we don't have access
        to token probabilities in API models, we use the discrete semantic entropy 
        approximation by sampling multiple completions.
        
        Args:
            prompt: Input question/prompt
            n: Number of responses to generate
            temperature: Sampling temperature
            
        Returns:
            List of generated response strings
        """
        try:
            # Paper specifies temperature=1.0 for sampling diversity
            response = await self.client.chat.completions.create(
                model=self.generation_model,
                messages=[{"role": "user", "content": prompt}],
                n=n,
                temperature=temperature,
                max_tokens=2048
            )
            
            return [choice.message.content.strip() for choice in response.choices if choice.message.content]
            
        except Exception as e:
            print(f"Error generating responses: {e}")
            return []
    
    async def check_entailment_structured(self, text1: str, text2: str, context: str) -> EntailmentResponse:
        """
        Check entailment between two texts using structured OpenAI output.
        
        Paper context (Section 3.1): Entailment is the core relation used to determine
        semantic equivalence. If text A entails text B, then B must be true whenever A is true.
        This is used to cluster responses by meaning rather than surface form.
        
        Args:
            text1: First text
            text2: Second text  
            context: The original question/context
            
        Returns:
            EntailmentResponse with structured result
        """
        system_prompt = """You are an expert in natural language inference performing STRICT entailment checking.
For entailment to hold, Text 1 must FULLY imply Text 2 - every piece of information in Text 2 must be derivable from Text 1.
Return your assessment with:
- relation: 'entailment' if Text 1 FULLY entails Text 2, 'contradiction' if they contradict, 'neutral' otherwise
- confidence: A score between 0 and 1
- reasoning: Brief explanation"""

        user_prompt = f"""Context Question: {context}

Text 1: {text1}
Text 2: {text2}

Does Text 1 semantically entail Text 2? That is, if Text 1 is true, must EVERYTHING in Text 2 also be true?"""

        try:
            # Use structured output with Pydantic model
            completion = await self.client.chat.completions.parse(
                model=self.entailment_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=2048,
                response_format=EntailmentResponse
            )
            
            return completion.choices[0].message.parsed
            
        except Exception as e:
            print(f"Error checking entailment: {e}")
            # Return neutral with low confidence as fallback
            return EntailmentResponse(
                relation=EntailmentRelation.NEUTRAL, 
                confidence=0.0, 
                reasoning="Error in entailment checking"
            )
    
    async def check_bidirectional_entailment(self, text1: str, text2: str, context: str) -> bool:
        """
        Check if two texts entail each other bidirectionally.
        
        Paper context (Section 3.1): Bidirectional entailment is used to determine semantic
        equivalence. Two responses are considered semantically equivalent if they entail each
        other - i.e., they express the same meaning even if worded differently.
        
        Args:
            text1: First text
            text2: Second text  
            context: The original question/context
            
        Returns:
            True if text1 entails text2 AND text2 entails text1
        """
        # Check both directions concurrently using asyncio.gather
        entailment_1_to_2, entailment_2_to_1 = await asyncio.gather(
            self.check_entailment_structured(text1, text2, context),
            self.check_entailment_structured(text2, text1, context)
        )
        
        # Both should be entailment with high confidence for strict semantic equivalence
        # Paper requires strict bidirectional entailment for clustering
        return (
            entailment_1_to_2.relation == EntailmentRelation.ENTAILMENT and
            entailment_2_to_1.relation == EntailmentRelation.ENTAILMENT and
            entailment_1_to_2.confidence > 0.7 and  # Increased threshold for stricter clustering
            entailment_2_to_1.confidence > 0.7
        )
    
    async def cluster_by_meaning(self, responses: List[str], context: str) -> List[SemanticCluster]:
        """
        Cluster responses by semantic meaning using bidirectional entailment.
        Implements Algorithm 1 from the paper with parallel processing for speed.
        
        Paper context (Algorithm 1): This is the core clustering algorithm. For each response,
        we check if it belongs to any existing cluster by testing bidirectional entailment
        with the cluster's representative (first element). If it matches, add to that cluster.
        If no match, create a new cluster. This groups responses by meaning, not surface form.
        
        Args:
            responses: List of response strings to cluster
            context: Original question/context
            
        Returns:
            List of SemanticCluster objects
        """
        if not responses:
            return []
        
        clusters: List[List[str]] = []
        
        for response in responses:
            # Check if this response belongs to any existing cluster
            placed = False
            
            if clusters:
                # Check against first item in each cluster concurrently
                # Paper: "transitivity of entailment means we only need to check against one representative"
                cluster_checks = []
                for cluster in clusters:
                    cluster_checks.append(
                        self.check_bidirectional_entailment(response, cluster[0], context)
                    )
                
                # Run all cluster checks in parallel for efficiency
                if cluster_checks:
                    cluster_results = await asyncio.gather(*cluster_checks)
                    
                    # Find first cluster that matches (responses can only belong to one cluster)
                    for i, matches in enumerate(cluster_results):
                        if matches and clusters[i]:
                            clusters[i].append(response)
                            placed = True
                            break
            
            # If not placed in any existing cluster, create a new one
            # This response represents a new semantic meaning not seen before
            if not placed:
                clusters.append([response])
        
        # Convert to SemanticCluster objects
        semantic_clusters = []
        for cluster_responses in clusters:
            cluster = SemanticCluster(responses=cluster_responses)
            semantic_clusters.append(cluster)
        
        return semantic_clusters

    def calculate_semantic_entropy(self, clusters: List[SemanticCluster], total_responses: int) -> float:
        """
        Calculate discrete semantic entropy from response clusters.
        Implements the discrete approximation from the paper.
        
        Paper context (Equation 2): Semantic entropy SE(x) = -Σ P(Ci|x) log P(Ci|x)
        where Ci are the semantic clusters and P(Ci|x) is the probability of cluster i.
        
        In the discrete approximation (Section 4.1), we estimate P(Ci|x) as:
        P(Ci|x) ≈ |Ci| / N where |Ci| is the number of responses in cluster i
        and N is the total number of sampled responses.
        
        High entropy indicates many different semantic meanings (likely confabulation).
        Low entropy indicates consistent semantic meaning (likely factual).
        
        Args:
            clusters: List of semantic clusters
            total_responses: Total number of responses
            
        Returns:
            Semantic entropy value
        """
        if not clusters or total_responses == 0:
            return 0.0
        
        # Calculate cluster probabilities: P(Ci|x) = |cluster_i| / total_responses
        entropy = 0.0
        
        for cluster in clusters:
            prob = cluster.size / total_responses
            if prob > 0:  # Avoid log(0)
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    async def detect_confabulation(
        self, 
        prompt: str, 
        num_samples: int = 10,
        entropy_threshold: float = 1.5
    ) -> ConfabulationResult:
        """
        Main pipeline for detecting confabulations using semantic entropy.
        Uses parallel processing for fast entailment checking.
        
        Paper context: This implements the full pipeline from the paper:
        1. Sample multiple responses from the model (Section 4.1)
        2. Cluster responses by semantic equivalence using Algorithm 1
        3. Calculate semantic entropy from clusters (Equation 2)
        4. Compare entropy to threshold to detect confabulation (Section 4.2)
        
        Args:
            prompt: Input question/prompt
            num_samples: Number of responses to generate for entropy calculation
            entropy_threshold: Threshold above which we flag as likely confabulation
            
        Returns:
            ConfabulationResult with complete analysis
        """
        # Step 1: Generate multiple responses at temperature=1.0 for entropy calculation
        # Paper: "We sample multiple outputs y1, ..., yn from the model"
        responses = await self.generate_responses(prompt, n=num_samples, temperature=1.0)
        
        if not responses:
            return ConfabulationResult(
                response="",
                semantic_entropy=0.0,
                likely_confabulation=True,
                num_clusters=0,
                num_responses=0,
                clusters=[],
                all_responses=[]
            )
        
        # Step 2: Cluster responses by meaning using Algorithm 1
        # Paper: "We cluster the outputs into semantic equivalence classes"
        clusters = await self.cluster_by_meaning(responses, prompt)
        
        # Step 3: Calculate semantic entropy from the clusters
        # Paper: "SE(x) = -Σ P(Ci|x) log P(Ci|x)"
        entropy = self.calculate_semantic_entropy(clusters, len(responses))
        
        # Step 4: Generate best response at low temperature for final answer
        # Not from paper - practical addition for returning a high-quality response
        best_responses = await self.generate_responses(prompt, n=1, temperature=0.1)
        best_response = best_responses[0] if best_responses else ""
        
        # Step 5: Determine confabulation likelihood based on entropy threshold
        # Paper shows confabulations have significantly higher entropy (often > 1.5)
        likely_confabulation = entropy > entropy_threshold
        
        return ConfabulationResult(
            response=best_response,
            semantic_entropy=entropy,
            likely_confabulation=likely_confabulation,
            num_clusters=len(clusters),
            num_responses=len(responses),
            clusters=clusters,
            all_responses=responses
        )


class HallucinationMonitorResponse(BaseModel):
    """Response object for hallucination monitoring"""
    score: float = Field(ge=0.0, le=100.0, description="Confabulation likelihood score (0-1)")
    semantic_entropy: float = Field(ge=0.0, description="Semantic entropy value")
    likely_confabulation: bool = Field(description="Whether this is likely a confabulation")
    num_clusters: int = Field(ge=0, description="Number of semantic clusters")


class HallucinationMonitor:
    """A hallucination monitor using semantic entropy from the Nature 2024 paper."""

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the hallucination monitor."""
        self.model = SemanticEntropyLLM(openai_api_key=openai_api_key)
    
    async def monitor_response(self, prompt: str, response: str) -> HallucinationMonitorResponse:
        """
        Monitor a model response for hallucination likelihood using semantic entropy.
        
        Args:
            prompt: The original input prompt
            response: The model's response to analyze
            
        Returns:
            HallucinationMonitorResponse with confabulation likelihood and details
        """
        # Analyze the prompt for confabulation likelihood
        result = await self.model.detect_confabulation(prompt, num_samples=10, entropy_threshold=1.5)
        
        # Return confabulation likelihood as float (0-1)
        score = 1.0 if result.likely_confabulation else 0.0
        
        return HallucinationMonitorResponse(
            score=score,
            semantic_entropy=result.semantic_entropy,
            likely_confabulation=result.likely_confabulation,
            num_clusters=result.num_clusters
        )
    
    def monitor_response_sync(self, prompt: str, response: str) -> HallucinationMonitorResponse:
        """
        Synchronous wrapper for monitor_response.
        
        Args:
            prompt: The original input prompt
            response: The model's response to analyze
            
        Returns:
            HallucinationMonitorResponse with confabulation likelihood and details
        """
        return asyncio.run(self.monitor_response(prompt, response))