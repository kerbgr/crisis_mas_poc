"""
Prompt Templates
Domain-specific prompt templates for different agent types and tasks
"""

from typing import Dict, Any


class PromptTemplates:
    """
    Collection of prompt templates for various crisis management tasks.
    """

    @staticmethod
    def get_scenario_assessment_prompt() -> str:
        """
        Get prompt template for scenario assessment.

        Returns:
            Prompt template string
        """
        return """You are a {expertise_domain} expert assessing a crisis situation.

**Crisis Type:** {scenario_type}
**Severity Level:** {severity}/1.0
**Affected Population:** {affected_population}

**Scenario Description:**
{scenario_description}

Please provide a professional assessment from your expertise perspective:

1. **Key Concerns:** What are the most critical issues from your domain's perspective?

2. **Risk Assessment:** What are the major risks and potential complications?

3. **Immediate Priorities:** What should be prioritized immediately?

4. **Resource Needs:** What specialized resources or expertise would be needed?

5. **Timeline Considerations:** What time-sensitive factors must be considered?

Please provide your assessment in a clear, structured format."""

    @staticmethod
    def get_action_evaluation_prompt() -> str:
        """
        Get prompt template for action evaluation.

        Returns:
            Prompt template string
        """
        return """You are a {expertise_domain} expert evaluating a proposed crisis response action.

**Scenario:** {scenario_type}
{scenario_description}

**Proposed Action:** {action_name}
{action_description}

Please evaluate this action from your professional perspective:

1. **Effectiveness:** How effective would this action be in addressing the crisis?
   - Rate: 0.0 (ineffective) to 1.0 (highly effective)
   - Rationale:

2. **Safety Considerations:** What safety risks or benefits does this action present?
   - Rate: 0.0 (unsafe) to 1.0 (very safe)
   - Key concerns:

3. **Implementation Feasibility:** How feasible is this action given current constraints?
   - Rate: 0.0 (infeasible) to 1.0 (highly feasible)
   - Challenges:

4. **Unintended Consequences:** What potential negative side effects should be considered?

5. **Recommendations:** What modifications or additional measures would you suggest?

Provide your evaluation with specific reasoning based on your expertise."""

    @staticmethod
    def get_action_justification_prompt() -> str:
        """
        Get prompt template for action justification.

        Returns:
            Prompt template string
        """
        return """You are a {expertise_domain} expert justifying a recommended crisis response action.

**Scenario:** {scenario_type}
{scenario_description}

**Recommended Action:** {action_name}
{action_description}

Please provide a detailed justification for why this action should be taken:

1. **Primary Benefits:** What are the main advantages of this approach?

2. **Evidence Base:** What evidence or precedents support this action?

3. **Risk Mitigation:** How does this action minimize risks?

4. **Resource Optimization:** How does this action make efficient use of available resources?

5. **Stakeholder Impact:** How will this action affect different stakeholders?

6. **Alternative Comparison:** Why is this action preferable to alternatives?

Provide a compelling, evidence-based justification that addresses both immediate and long-term considerations."""

    @staticmethod
    def get_consensus_facilitation_prompt() -> str:
        """
        Get prompt template for consensus facilitation.

        Returns:
            Prompt template string
        """
        return """You are a crisis management coordinator facilitating consensus among expert advisors.

**Scenario:**
{scenario_description}

**Expert Proposals:**
{num_experts} experts have provided the following recommendations:

{proposals_summary}

As the coordinator, please:

1. **Analyze Disagreements:** Identify the key points of disagreement and their underlying reasons.

2. **Find Common Ground:** What aspects do most experts agree on?

3. **Synthesize Recommendations:** Can elements from different proposals be combined into a stronger approach?

4. **Assess Trade-offs:** What are the trade-offs between different expert recommendations?

5. **Recommend Path Forward:** Based on all expert input, what course of action would best serve the crisis response?

6. **Build Consensus:** How can dissenting experts' concerns be addressed or mitigated?

Provide a balanced synthesis that respects diverse expertise while working toward an actionable decision."""

    @staticmethod
    def get_risk_analysis_prompt() -> str:
        """
        Get prompt template for risk analysis.

        Returns:
            Prompt template string
        """
        return """You are a {expertise_domain} expert conducting risk analysis for a crisis response plan.

**Scenario:** {scenario_type}
{scenario_description}

**Proposed Plan:** {action_name}

Please conduct a comprehensive risk analysis:

1. **High-Probability Risks:** What risks are most likely to occur?
   - List with probability estimates

2. **High-Impact Risks:** What risks would have the most severe consequences?
   - List with impact assessments

3. **Cascade Effects:** What secondary or tertiary effects might occur?

4. **Mitigation Strategies:** For each major risk, what mitigation measures are recommended?

5. **Monitoring Indicators:** What early warning signs should be monitored?

6. **Contingency Planning:** What backup plans should be prepared?

Provide a structured risk analysis that can inform decision-making and contingency planning."""

    @staticmethod
    def get_multi_criteria_comparison_prompt() -> str:
        """
        Get prompt template for multi-criteria comparison.

        Returns:
            Prompt template string
        """
        return """You are a decision analysis expert comparing multiple crisis response options.

**Scenario:**
{scenario_description}

**Options to Compare:**
{options_list}

**Decision Criteria:**
{criteria_list}

Please provide a structured comparison:

1. **Criteria Performance:** How does each option perform on each criterion?

2. **Trade-off Analysis:** What are the key trade-offs between options?

3. **Sensitivity:** Which decision criteria are most critical to the final choice?

4. **Robustness:** Which option performs reasonably well across all criteria?

5. **Context Considerations:** What scenario-specific factors should influence the choice?

6. **Recommendation:** Based on balanced analysis, which option(s) should be prioritized?

Provide an objective, analytical comparison that helps decision-makers understand the strengths and weaknesses of each option."""

    @staticmethod
    def format_prompt(template: str, **kwargs) -> str:
        """
        Format a prompt template with provided values.

        Args:
            template: Prompt template string
            **kwargs: Values to fill in the template

        Returns:
            Formatted prompt
        """
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required template parameter: {e}")

    @staticmethod
    def get_all_templates() -> Dict[str, str]:
        """
        Get all available prompt templates.

        Returns:
            Dictionary mapping template names to templates
        """
        return {
            'scenario_assessment': PromptTemplates.get_scenario_assessment_prompt(),
            'action_evaluation': PromptTemplates.get_action_evaluation_prompt(),
            'action_justification': PromptTemplates.get_action_justification_prompt(),
            'consensus_facilitation': PromptTemplates.get_consensus_facilitation_prompt(),
            'risk_analysis': PromptTemplates.get_risk_analysis_prompt(),
            'multi_criteria_comparison': PromptTemplates.get_multi_criteria_comparison_prompt()
        }
