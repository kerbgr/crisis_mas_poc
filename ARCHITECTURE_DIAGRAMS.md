# Crisis Management MAS - Architecture Diagrams

**Comprehensive Mermaid diagrams for system architecture and data flow**

---

## 1. Overall System Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        CLI[Command Line Interface]
        JSON_IN[JSON Input Files]
        JSON_OUT[JSON Output Results]
        VIZ[Visualization Generator]
    end

    subgraph "Coordination Layer"
        COORD[CoordinatorAgent]
        ORCH[Orchestration Logic]
        CONS_BUILD[Consensus Builder]
    end

    subgraph "Agent Layer"
        BA[BaseAgent]
        EA1[Medical Expert]
        EA2[Logistics Expert]
        EA3[Safety Expert]
        EA4[Environmental Expert]
        RT[ReliabilityTracker]
        PROFILES[Agent Profiles JSON]
    end

    subgraph "Decision Framework Layer"
        ER[Evidential Reasoning]
        GAT[GAT Aggregator<br/>9D Features]
        MCDA[MCDA Engine<br/>TOPSIS]
        CONSENSUS[Consensus Model<br/>Cosine Similarity]
    end

    subgraph "LLM Integration Layer"
        LLM_INT[LLM Interface]
        CLAUDE[Claude Client]
        OPENAI[OpenAI Client]
        LMSTUDIO[LM Studio Client]
        PROMPTS[Prompt Templates]
    end

    subgraph "Evaluation & Utilities Layer"
        METRICS[Metrics Calculator]
        VIS[Visualizations]
        VALID[Validation]
        CONFIG[Configuration]
    end

    CLI --> COORD
    JSON_IN --> COORD
    COORD --> ORCH
    ORCH --> EA1 & EA2 & EA3 & EA4
    EA1 & EA2 & EA3 & EA4 -.inherits.-> BA
    BA --> RT
    BA --> PROFILES

    ORCH --> ER & GAT
    ER --> CONSENSUS
    GAT --> CONSENSUS
    CONSENSUS --> CONS_BUILD
    CONS_BUILD --> MCDA

    EA1 & EA2 & EA3 & EA4 --> LLM_INT
    LLM_INT --> CLAUDE & OPENAI & LMSTUDIO
    LLM_INT --> PROMPTS

    MCDA --> METRICS
    METRICS --> VIS
    VIS --> VIZ
    VIZ --> JSON_OUT

    VALID -.validates.-> JSON_IN
    CONFIG -.configures.-> COORD & LLM_INT

    style COORD fill:#ff9999
    style GAT fill:#99ccff
    style RT fill:#99ff99
    style LLM_INT fill:#ffcc99
```

---

## 2. Multi-Agent Decision Flow

```mermaid
sequenceDiagram
    participant User
    participant Coordinator
    participant Medical as Medical Expert
    participant Logistics as Logistics Expert
    participant Safety as Safety Expert
    participant Environment as Environment Expert
    participant GAT as GAT Aggregator
    participant MCDA as MCDA Engine
    participant Consensus

    User->>Coordinator: Submit Crisis Scenario

    Note over Coordinator: Distribute to Experts

    par Parallel Agent Evaluation
        Coordinator->>Medical: evaluate_scenario()
        Medical->>Medical: LLM Reasoning
        Medical->>Medical: Generate Belief Distribution
        Medical-->>Coordinator: {belief, confidence, reasoning}

        Coordinator->>Logistics: evaluate_scenario()
        Logistics->>Logistics: LLM Reasoning
        Logistics->>Logistics: Generate Belief Distribution
        Logistics-->>Coordinator: {belief, confidence, reasoning}

        Coordinator->>Safety: evaluate_scenario()
        Safety->>Safety: LLM Reasoning
        Safety->>Safety: Generate Belief Distribution
        Safety-->>Coordinator: {belief, confidence, reasoning}

        Coordinator->>Environment: evaluate_scenario()
        Environment->>Environment: LLM Reasoning
        Environment->>Environment: Generate Belief Distribution
        Environment-->>Coordinator: {belief, confidence, reasoning}
    end

    Note over Coordinator: Aggregate Beliefs

    Coordinator->>GAT: aggregate_beliefs_with_gat()
    GAT->>GAT: Extract 9D Features
    GAT->>GAT: Compute Attention Weights
    GAT->>GAT: Aggregate Beliefs
    GAT-->>Coordinator: Aggregated Distribution + Weights

    Coordinator->>Consensus: analyze_consensus()
    Consensus->>Consensus: Calculate Similarity
    Consensus->>Consensus: Detect Conflicts

    alt Consensus Reached
        Consensus-->>Coordinator: Consensus Achieved
        Coordinator->>MCDA: rank_alternatives()
        MCDA-->>Coordinator: Ranked Alternatives
        Coordinator-->>User: Final Decision + Explanation
    else Conflict Detected
        Consensus-->>Coordinator: Conflicts Found
        Coordinator->>Coordinator: Iterative Refinement
        Coordinator->>Medical: Refine Assessment
        Note over Coordinator: Repeat until consensus or max iterations
    end
```

---

## 3. GAT Feature Extraction and Aggregation

```mermaid
flowchart TB
    START[Agent Assessments + Scenario] --> EXTRACT[Extract Features per Agent]

    subgraph "9-Dimensional Feature Extraction"
        EXTRACT --> F1[F1: Confidence Score]
        EXTRACT --> F2[F2: Belief Certainty<br/>inverse entropy]
        EXTRACT --> F3[F3: Expertise Relevance<br/>to scenario]
        EXTRACT --> F4[F4: Risk Tolerance]
        EXTRACT --> F5[F5: Severity Awareness]
        EXTRACT --> F6[F6: Top Choice Strength<br/>margin between top 2]
        EXTRACT --> F7[F7: Thoroughness<br/>number of concerns]
        EXTRACT --> F8[F8: Reasoning Quality<br/>length proxy]
        EXTRACT --> F9[F9: Historical Reliability<br/>from ReliabilityTracker ⭐]
    end

    F1 & F2 & F3 & F4 & F5 & F6 & F7 & F8 & F9 --> VECTOR[Feature Vector fi]

    VECTOR --> BUILD_ADJ[Build Adjacency Matrix<br/>Trust Relationships]

    BUILD_ADJ --> ATTENTION[Compute Attention Coefficients]

    subgraph "Multi-Head Attention (H=4)"
        ATTENTION --> HEAD1[Head 1<br/>Compute αij¹]
        ATTENTION --> HEAD2[Head 2<br/>Compute αij²]
        ATTENTION --> HEAD3[Head 3<br/>Compute αij³]
        ATTENTION --> HEAD4[Head 4<br/>Compute αij⁴]

        HEAD1 & HEAD2 & HEAD3 & HEAD4 --> AVG[Average Attention<br/>αij = mean over heads]
    end

    AVG --> SOFTMAX[Softmax Normalization<br/>per row]

    SOFTMAX --> AGGREGATE[Weighted Aggregation<br/>of Belief Distributions]

    AGGREGATE --> OUTPUT[Aggregated Beliefs +<br/>Attention Weights +<br/>Confidence +<br/>Explanation]

    style F9 fill:#99ff99
    style AGGREGATE fill:#ff9999
```

---

## 4. Historical Reliability Tracking Workflow

```mermaid
stateDiagram-v2
    [*] --> AssessmentMade: Agent makes prediction

    AssessmentMade --> Recorded: record_assessment(id, type, prediction, confidence)

    state Recorded {
        [*] --> StoredInHistory
        StoredInHistory --> AddedToRecentWindow
        AddedToRecentWindow --> [*]
    }

    Recorded --> WaitingForOutcome: Stored in assessment_history

    WaitingForOutcome --> OutcomeReceived: Actual outcome occurs

    OutcomeReceived --> CalculateAccuracy: update_assessment_outcome(id, outcome)

    state CalculateAccuracy {
        [*] --> ProbabilityScore: Belief for actual outcome (40%)
        [*] --> RankAccuracy: Was top choice correct? (30%)
        [*] --> MarginScore: Confidence appropriateness (30%)

        ProbabilityScore --> CombineScores
        RankAccuracy --> CombineScores
        MarginScore --> CombineScores

        CombineScores --> FinalAccuracy: accuracy ∈ [0,1]
        FinalAccuracy --> [*]
    }

    CalculateAccuracy --> UpdateMetrics

    state UpdateMetrics {
        [*] --> OverallReliability: Lifetime with temporal decay<br/>γ^(T-t)
        [*] --> RecentReliability: Sliding window (last 10)
        [*] --> ConsistencyScore: Inverse variance
        [*] --> DomainReliability: Per crisis type

        OverallReliability --> [*]
        RecentReliability --> [*]
        ConsistencyScore --> [*]
        DomainReliability --> [*]
    }

    UpdateMetrics --> UpdateConfidence: Update agent confidence_level

    UpdateConfidence --> UsedInGAT: Feature 9 in next aggregation

    UsedInGAT --> [*]

    note right of CalculateAccuracy
        Three-component accuracy:
        • Probability: belief[actual]
        • Rank: correct top choice?
        • Margin: confidence match?
    end note

    note right of UsedInGAT
        Agents with high reliability
        receive higher attention weights
        in future aggregations
    end note
```

---

## 5. LLM Integration Architecture

```mermaid
graph LR
    subgraph "Agent Layer"
        EA[Expert Agent]
    end

    subgraph "LLM Integration Layer"
        INT[LLM Interface]
        PT[Prompt Templates]

        subgraph "Provider Clients"
            CLAUDE[Claude Client<br/>Anthropic API]
            OPENAI[OpenAI Client<br/>GPT-4/3.5]
            LMS[LM Studio Client<br/>Local Models]
        end

        RETRY[Retry Logic<br/>Exponential Backoff]
        PARSE[Response Parser]
        CACHE[Response Cache<br/>15-min TTL]
    end

    EA -->|generate_assessment| INT
    INT --> PT
    PT -->|Crisis Scenario Prompt| PROVIDER{Provider Selection}

    PROVIDER -->|provider=claude| CLAUDE
    PROVIDER -->|provider=openai| OPENAI
    PROVIDER -->|provider=lmstudio| LMS

    CLAUDE --> RETRY
    OPENAI --> RETRY
    LMS --> RETRY

    RETRY -->|Success| CACHE
    RETRY -->|Failure| RETRY
    RETRY -->|Max Retries| ERROR[Error Handler]

    CACHE --> PARSE
    PARSE --> STRUCTURE[Structured Response:<br/>belief_distribution<br/>confidence<br/>reasoning<br/>key_concerns]

    STRUCTURE -->|Return| EA
    ERROR -->|Fallback| RULE[Rule-Based Fallback]
    RULE --> EA

    style CLAUDE fill:#ff9999
    style OPENAI fill:#99ccff
    style LMS fill:#99ff99
    style CACHE fill:#ffcc99
```

---

## 6. Decision Framework Component Interactions

```mermaid
graph TB
    subgraph "Input"
        SCENARIO[Crisis Scenario]
        ALTERNATIVES[Alternative Actions]
        AGENT_ASSESS[Agent Assessments]
    end

    subgraph "Evidential Reasoning Path"
        ER_AGG[ER: Weighted Averaging]
        ER_NORM[Normalize Weights]
        ER_COMBINE[Combine Beliefs]
        ER_CONF[Calculate Confidence<br/>Entropy-based]

        ER_NORM --> ER_COMBINE
        ER_COMBINE --> ER_CONF
    end

    subgraph "GAT Path"
        GAT_FEAT[Extract 9D Features]
        GAT_ADJ[Build Adjacency Matrix]
        GAT_ATT[Compute Attention<br/>Multi-Head]
        GAT_AGG[Aggregate with Weights]

        GAT_FEAT --> GAT_ATT
        GAT_ADJ --> GAT_ATT
        GAT_ATT --> GAT_AGG
    end

    subgraph "Consensus Layer"
        COSINE[Cosine Similarity]
        DETECT[Detect Conflicts]
        RESOLVE[Resolution Suggestions]

        COSINE --> DETECT
        DETECT -->|Conflicts Found| RESOLVE
        DETECT -->|Consensus| PROCEED
    end

    subgraph "MCDA Layer"
        NORM[Normalize Criteria]
        SCORE[Calculate Scores<br/>per Alternative]
        RANK[Rank by Score]
        SENS[Sensitivity Analysis]

        NORM --> SCORE
        SCORE --> RANK
        RANK --> SENS
    end

    subgraph "Output"
        DECISION[Final Decision]
        EXPLAIN[Explanation]
        VIS[Visualizations]
        METRICS[Performance Metrics]
    end

    AGENT_ASSESS --> ER_AGG
    AGENT_ASSESS --> GAT_FEAT
    SCENARIO --> GAT_FEAT

    ER_CONF --> COSINE
    GAT_AGG --> COSINE

    PROCEED --> NORM
    ALTERNATIVES --> NORM

    RANK --> DECISION
    SENS --> EXPLAIN
    GAT_ATT --> EXPLAIN

    DECISION --> VIS
    EXPLAIN --> VIS

    VIS --> METRICS

    style ER_AGG fill:#ffcccc
    style GAT_AGG fill:#ccddff
    style DECISION fill:#99ff99
```

---

## 7. Reliability Score Calculation Flow

```mermaid
flowchart TB
    START[Historical Assessments] --> FILTER{Filter by Mode}

    FILTER -->|overall| ALL[All Evaluated Assessments]
    FILTER -->|recent| WINDOW[Last 10 Assessments]
    FILTER -->|consistent| VARIANCE[Calculate Variance]
    FILTER -->|domain| DOMAIN[Filter by Crisis Type]

    ALL --> TEMPORAL[Apply Temporal Decay]

    subgraph "Temporal Decay Weighting"
        TEMPORAL --> CALC_DECAY[wt = γ^T-t<br/>γ = 0.95]
        CALC_DECAY --> WEIGHT_ACC[weighted_accuracy_sum]
        CALC_DECAY --> WEIGHT_TOTAL[weight_total]
        WEIGHT_ACC --> DIVIDE[reliability = sum / total]
    end

    WINDOW --> MEAN[Mean of Recent Scores]

    VARIANCE --> CALC_VAR[var = σ²]
    CALC_VAR --> CONSISTENCY[consistency = 1 / 1+var]

    DOMAIN --> GROUP[Group by Crisis Type]
    GROUP --> DOMAIN_MEAN[Mean per Type]

    DIVIDE --> OUTPUT[Overall Reliability Score]
    MEAN --> OUTPUT2[Recent Reliability Score]
    CONSISTENCY --> OUTPUT3[Consistency Score]
    DOMAIN_MEAN --> OUTPUT4[Domain Reliability Map]

    OUTPUT --> USE[Use in GAT Feature 9]
    OUTPUT2 --> USE
    OUTPUT3 --> USE
    OUTPUT4 --> USE

    style TEMPORAL fill:#ffcc99
    style USE fill:#99ff99
```

---

## 8. Consensus Building Process

```mermaid
flowchart TB
    START[Agent Belief Distributions] --> VECTORIZE[Convert to Vectors]

    VECTORIZE --> PAIRWISE[Compute Pairwise<br/>Cosine Similarity]

    subgraph "Similarity Calculation"
        PAIRWISE --> DOT[Dot Product: A·B]
        PAIRWISE --> NORM_A[Magnitude: ||A||]
        PAIRWISE --> NORM_B[Magnitude: ||B||]

        DOT --> COSINE_CALC[cos θ = A·B / ||A||×||B||]
        NORM_A --> COSINE_CALC
        NORM_B --> COSINE_CALC
    end

    COSINE_CALC --> AVG_SIM[Average All Pairs<br/>consensus_level]

    AVG_SIM --> THRESHOLD{consensus_level<br/>≥ threshold?}

    THRESHOLD -->|Yes| CONSENSUS_OK[Consensus Achieved ✓]
    THRESHOLD -->|No| DETECT_CONFLICTS[Detect Conflicts]

    DETECT_CONFLICTS --> FIND_PAIRS[Find Disagreeing Pairs]
    FIND_PAIRS --> TOP_CHOICES[Compare Top Choices]
    TOP_CHOICES --> CONFLICT_SCORE[Calculate Conflict Severity]

    CONFLICT_SCORE --> CLASSIFY{Severity}

    CLASSIFY -->|Low < 0.3| LOW[Low Severity<br/>Weighted Voting]
    CLASSIFY -->|0.3 ≤ Moderate < 0.6| MED[Moderate Severity<br/>Find Compromise]
    CLASSIFY -->|High ≥ 0.6| HIGH[High Severity<br/>Human Escalation]

    LOW --> SUGGEST[Resolution Suggestions]
    MED --> SUGGEST
    HIGH --> SUGGEST

    SUGGEST --> COMPROMISE[Find Compromise Alternatives<br/>combined_score = mean belief]

    CONSENSUS_OK --> PROCEED[Proceed to MCDA]
    COMPROMISE --> ITERATE[Iterative Refinement]
    ITERATE --> PAIRWISE

    style CONSENSUS_OK fill:#99ff99
    style HIGH fill:#ff9999
    style MED fill:#ffcc99
```

---

## 9. Complete System Data Flow (End-to-End)

```mermaid
graph TB
    USER[User Input:<br/>Scenario + Alternatives]

    subgraph "Phase 1: Distribution"
        COORD1[Coordinator receives request]
        DIST[Distribute to all agents]
    end

    subgraph "Phase 2: Individual Assessment"
        AG1[Medical Expert]
        AG2[Logistics Expert]
        AG3[Safety Expert]
        AG4[Environment Expert]

        LLM1[LLM: Claude/OpenAI/<br/>LM Studio]

        AG1 & AG2 & AG3 & AG4 --> LLM1
        LLM1 --> ASSESS[Generate Assessments:<br/>belief_distribution<br/>confidence<br/>reasoning]

        ASSESS --> RT1[Record Assessment<br/>in ReliabilityTracker]
    end

    subgraph "Phase 3: Aggregation"
        RT1 --> COLLECT[Collect All Assessments]

        COLLECT --> GAT1[GAT Aggregation]
        GAT1 --> FEAT[Extract 9D Features<br/>incl. reliability]
        FEAT --> ATT[Compute Attention]
        ATT --> AGG[Weighted Aggregation]

        AGG --> CONS1[Consensus Check]
        CONS1 --> COS[Cosine Similarity]
        COS -->|consensus_level| DECISION{Threshold Met?}
    end

    subgraph "Phase 4: Decision Making"
        DECISION -->|Yes| MCDA1[MCDA Ranking]
        DECISION -->|No| CONFLICT[Conflict Resolution]

        CONFLICT --> REFINE[Iterative Refinement]
        REFINE -.retry.-> COLLECT

        MCDA1 --> TOPSIS[TOPSIS Evaluation]
        TOPSIS --> RANK[Ranked Alternatives]
    end

    subgraph "Phase 5: Output"
        RANK --> EXPLAIN[Generate Explanation]
        EXPLAIN --> VIS1[Create Visualizations]
        VIS1 --> METRICS1[Calculate Metrics]
        METRICS1 --> OUTPUT[Final Output:<br/>Decision + Reasoning +<br/>Metrics + Visualizations]
    end

    subgraph "Phase 6: Learning"
        OUTPUT --> WAIT[Wait for Actual Outcome]
        WAIT --> ACTUAL[Actual Outcome Received]
        ACTUAL --> UPDATE[Update Reliability:<br/>update_assessment_outcome]
        UPDATE --> RT2[ReliabilityTracker<br/>updates metrics]
        RT2 -.future assessments.-> FEAT
    end

    USER --> COORD1
    COORD1 --> DIST
    DIST --> AG1 & AG2 & AG3 & AG4
    OUTPUT --> USER

    style USER fill:#e1f5ff
    style OUTPUT fill:#99ff99
    style RT2 fill:#ffcc99
    style GAT1 fill:#ccddff
```

---

## 10. Class Diagram (Core Components)

```mermaid
classDiagram
    class BaseAgent {
        <<abstract>>
        +agent_id: str
        +name: str
        +expertise: str
        +confidence_level: float
        +reliability_tracker: ReliabilityTracker
        +evaluate_scenario()
        +propose_action()
        +get_reliability_score()
        +record_assessment()
        +update_assessment_outcome()
    }

    class ExpertAgent {
        +llm_client: LLMClient
        +profile: Dict
        +evaluate_scenario()
        +propose_action()
        -_generate_llm_assessment()
    }

    class CoordinatorAgent {
        +expert_agents: List[ExpertAgent]
        +aggregator: GATAggregator
        +consensus_model: ConsensusModel
        +orchestrate_decision()
        +aggregate_beliefs()
        +build_consensus()
    }

    class ReliabilityTracker {
        +agent_id: str
        +assessment_history: List
        +metrics: ReliabilityMetrics
        +record_assessment()
        +update_assessment_outcome()
        +get_reliability_score()
        -_calculate_accuracy()
        -_update_reliability_metrics()
    }

    class ReliabilityMetrics {
        +overall_reliability: float
        +recent_reliability: float
        +consistency_score: float
        +domain_reliability: Dict
        +total_assessments: int
    }

    class GATAggregator {
        +num_heads: int
        +attention_layers: List
        +extract_agent_features()
        +compute_attention_coefficients()
        +aggregate_beliefs_with_gat()
    }

    class GraphAttentionLayer {
        +feature_dim: int = 9
        +attention_heads: int
        +extract_agent_features()
        +compute_attention_coefficients()
    }

    class EvidentialReasoning {
        +combine_beliefs()
        +normalize_distribution()
        +calculate_confidence()
    }

    class MCDAEngine {
        +criteria_config: Dict
        +rank_alternatives()
        +normalize_score()
        +calculate_weighted_score()
        +sensitivity_analysis()
    }

    class ConsensusModel {
        +consensus_threshold: float
        +calculate_consensus_level()
        +detect_conflicts()
        +suggest_resolution()
    }

    class LLMClient {
        <<interface>>
        +generate_assessment()
        +generate_with_retry()
    }

    class ClaudeClient {
        +client: Anthropic
        +generate_assessment()
    }

    class OpenAIClient {
        +client: OpenAI
        +generate_assessment()
    }

    class LMStudioClient {
        +base_url: str
        +generate_assessment()
    }

    BaseAgent <|-- ExpertAgent
    BaseAgent <|-- CoordinatorAgent
    BaseAgent *-- ReliabilityTracker
    ReliabilityTracker *-- ReliabilityMetrics

    CoordinatorAgent o-- ExpertAgent
    CoordinatorAgent *-- GATAggregator
    CoordinatorAgent *-- ConsensusModel
    CoordinatorAgent *-- MCDAEngine
    CoordinatorAgent *-- EvidentialReasoning

    GATAggregator *-- GraphAttentionLayer

    ExpertAgent o-- LLMClient
    LLMClient <|-- ClaudeClient
    LLMClient <|-- OpenAIClient
    LLMClient <|-- LMStudioClient
```

---

## Diagram Legend

| Symbol | Meaning |
|--------|---------|
| `-->` | Data flow / dependency |
| `-.->` | Dashed: Optional / fallback flow |
| `==>` | Thick: Primary flow path |
| `o--` | Composition |
| `*--` | Aggregation |
| `<|--` | Inheritance |
| Colored boxes | Different system layers |

## Usage in Documentation

These diagrams can be embedded in:
- **README.md** - Main documentation
- **Thesis document** - System architecture chapter
- **Presentations** - Defense slides
- **Academic papers** - System description sections

All diagrams are in Mermaid format and will render automatically on GitHub, GitLab, and most Markdown viewers.

---

**Generated:** 2025-11-07
**System:** Crisis Management Multi-Agent System
**Version:** 1.0
