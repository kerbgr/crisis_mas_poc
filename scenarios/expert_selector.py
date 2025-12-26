"""
Expert Selector - Intelligent Agent Selection Based on Scenario Characteristics

This module provides automatic expert agent selection based on scenario metadata,
eliminating the need for manual expert selection for each crisis scenario.

The ExpertSelector analyzes scenario characteristics and automatically determines
which of the 11 available expert agents should be engaged based on:
- Crisis type and subtypes
- Severity and scope
- Affected domains and infrastructure
- Command structure requirements
- Multi-jurisdictional coordination needs

Author: Crisis MAS Development Team
Version: 1.0
Last Updated: November 2025
"""

import logging
from typing import Dict, List, Any, Set

logger = logging.getLogger(__name__)


class ExpertSelector:
    """
    Intelligent expert selection based on scenario characteristics.

    Uses rule-based matching to determine which expert agents should be
    engaged for a given crisis scenario. Ensures minimum viable expert
    team while avoiding unnecessary overhead.

    Example:
        selector = ExpertSelector()
        scenario = load_scenario('scenarios/coastal_flood.json')
        agent_ids = selector.select_experts(scenario)
        # Returns: ['agent_meteorologist', 'medical_expert_01', 'logistics_expert_01',
        #           'coastguard_onscene_01', 'coastguard_national_01', ...]
    """

    # Minimum experts always included (core team)
    CORE_EXPERTS = {
        'agent_meteorologist': 'Meteorologist',
        'logistics_expert_01': 'Logistics Coordinator',
        'medical_expert_01': 'Medical Expert'
    }

    # Expert selection rules: maps expert roles to selection criteria
    EXPERT_SELECTION_RULES = {
        # Core experts (almost always needed)
        'meteorologist': {
            'agent_id': 'agent_meteorologist',
            'crisis_types': ['flood', 'hurricane', 'storm', 'wildfire', 'tornado', 'earthquake', 'tsunami'],
            'domains': ['weather_environment'],
            'always_include': True,  # Always valuable for situational awareness
            'description': 'Weather/environmental analysis for crisis planning'
        },
        'logistics': {
            'agent_id': 'logistics_expert_01',
            'always_include': True,  # Resource allocation always needed
            'severity_threshold': 0.0,  # Include for all severity levels
            'description': 'Supply chain and resource coordination'
        },
        'medical': {
            'agent_id': 'medical_expert_01',
            'domains': ['medical_health'],
            'affected_populations_threshold': 50,  # Include if >50 people affected
            'always_include': True,  # Public health always a consideration
            'description': 'Emergency medicine and public health response'
        },

        # Emergency Communications
        'psap_commander': {
            'agent_id': 'psap_commander_01',
            'command_structure': ['tactical'],
            'domains': ['emergency_communications'],
            'multi_jurisdictional': True,
            'severity_threshold': 0.5,
            'description': 'Emergency dispatch and multi-agency coordination'
        },

        # Police (tactical + strategic)
        'police_onscene': {
            'agent_id': 'police_onscene_01',
            'crisis_types': ['civil_unrest', 'terrorist', 'evacuation', 'flood', 'earthquake', 'fire'],
            'crisis_subtypes': ['evacuation', 'crowd_control', 'public_order'],
            'command_structure': ['tactical'],
            'domains': ['law_enforcement', 'public_order', 'security'],
            'severity_threshold': 0.5,
            'description': 'Tactical law enforcement and scene security'
        },
        'police_regional': {
            'agent_id': 'police_regional_01',
            'geographic_scope': ['regional', 'national'],
            'command_structure': ['strategic'],
            'multi_jurisdictional': True,
            'severity_threshold': 0.6,
            'affected_populations_threshold': 5000,
            'description': 'Strategic police coordination across jurisdictions'
        },

        # Fire/Rescue (tactical + strategic)
        'fire_onscene': {
            'agent_id': 'fire_onscene_01',
            'crisis_types': ['fire', 'wildfire', 'explosion', 'hazmat', 'earthquake', 'building_collapse'],
            'crisis_subtypes': ['structural_damage', 'rescue', 'hazmat'],
            'command_structure': ['tactical'],
            'domains': ['fire_rescue', 'hazmat', 'search_rescue'],
            'severity_threshold': 0.4,
            'description': 'Tactical fire suppression and rescue operations'
        },
        'fire_regional': {
            'agent_id': 'fire_regional_01',
            'crisis_types': ['wildfire', 'multiple_fires', 'large_fire'],
            'geographic_scope': ['regional', 'national'],
            'command_structure': ['strategic'],
            'duration_hours_threshold': 12,  # Long-duration incidents
            'severity_threshold': 0.7,
            'description': 'Strategic fire service deployment and mutual aid'
        },

        # Medical Infrastructure
        'medical_infrastructure': {
            'agent_id': 'medical_infrastructure_01',
            'domains': ['medical_health'],
            'infrastructure_systems': ['hospitals', 'healthcare'],
            'affected_populations_threshold': 1000,
            'severity_threshold': 0.6,
            'crisis_types': ['pandemic', 'mass_casualty', 'chemical', 'biological'],
            'description': 'Hospital capacity and healthcare system coordination'
        },

        # Coast Guard (tactical + strategic)
        'coastguard_onscene': {
            'agent_id': 'coastguard_onscene_01',
            'crisis_types': ['flood', 'hurricane', 'maritime', 'tsunami', 'coastal_flood', 'storm_surge'],
            'crisis_subtypes': ['coastal', 'maritime', 'offshore'],
            'command_structure': ['tactical'],
            'domains': ['maritime_coastal', 'search_rescue'],
            'geographic_location': ['coastal', 'maritime'],
            'description': 'Maritime rescue and coastal evacuation operations'
        },
        'coastguard_national': {
            'agent_id': 'coastguard_national_01',
            'crisis_types': ['hurricane', 'tsunami', 'oil_spill', 'maritime_disaster'],
            'geographic_scope': ['national', 'regional'],
            'domains': ['maritime_coastal'],
            'command_structure': ['strategic'],
            'severity_threshold': 0.7,
            'affected_populations_threshold': 10000,
            'description': 'Strategic maritime coordination and port security'
        }
    }

    def __init__(self, min_experts: int = 3, max_experts: int = 11, verbose: bool = False):
        """
        Initialize ExpertSelector.

        Args:
            min_experts: Minimum number of experts to select (default: 3 core)
            max_experts: Maximum number of experts to select (default: 11 all)
            verbose: Enable verbose logging of selection process
        """
        self.min_experts = min_experts
        self.max_experts = max_experts
        self.verbose = verbose

    def select_experts(self, scenario: Dict[str, Any]) -> List[str]:
        """
        Automatically select expert agents based on scenario characteristics.

        Args:
            scenario: Scenario dictionary with 'expert_selection' metadata

        Returns:
            List of agent IDs that should be engaged

        Example:
            >>> scenario = {
            ...     'expert_selection': {
            ...         'crisis_type': 'flood',
            ...         'crisis_subtypes': ['coastal', 'evacuation'],
            ...         'severity': 0.85,
            ...         'affected_domains': ['maritime_coastal', 'law_enforcement']
            ...     }
            ... }
            >>> selector.select_experts(scenario)
            ['agent_meteorologist', 'logistics_expert_01', 'medical_expert_01',
             'coastguard_onscene_01', 'police_onscene_01']
        """
        expert_meta = scenario.get('expert_selection', {})

        if not expert_meta:
            logger.warning("No 'expert_selection' metadata found in scenario. Using default 3 core experts.")
            return list(self.CORE_EXPERTS.keys())

        selected: Set[str] = set()
        selection_reasons = {}

        # Evaluate each expert against selection rules
        for expert_key, rules in self.EXPERT_SELECTION_RULES.items():
            match_score, reasons = self._evaluate_expert(expert_meta, rules)

            if match_score > 0 or rules.get('always_include', False):
                agent_id = rules['agent_id']
                selected.add(agent_id)
                selection_reasons[agent_id] = {
                    'score': match_score,
                    'reasons': reasons,
                    'description': rules.get('description', '')
                }

        # Ensure minimum experts (core team)
        if len(selected) < self.min_experts:
            for agent_id in self.CORE_EXPERTS.keys():
                selected.add(agent_id)

        # Cap at maximum experts
        if len(selected) > self.max_experts:
            # Keep highest scoring experts
            sorted_experts = sorted(
                selection_reasons.items(),
                key=lambda x: x[1]['score'],
                reverse=True
            )
            selected = set([e[0] for e in sorted_experts[:self.max_experts]])

        selected_list = list(selected)

        if self.verbose:
            self._log_selection_details(scenario, selected_list, selection_reasons)
        else:
            logger.info(f"Auto-selected {len(selected_list)} experts based on scenario metadata")

        return selected_list

    def _evaluate_expert(self, scenario_meta: Dict, rules: Dict) -> tuple[int, List[str]]:
        """
        Evaluate if scenario metadata matches expert selection rules.

        Args:
            scenario_meta: Scenario expert_selection metadata
            rules: Expert selection rules

        Returns:
            Tuple of (match_score, list_of_reasons)
        """
        score = 0
        reasons = []

        # Always include
        if rules.get('always_include'):
            return (10, ['Core expert - always included'])

        # Crisis type matching
        if 'crisis_types' in rules:
            crisis_type = scenario_meta.get('crisis_type', '').lower()
            if crisis_type in rules['crisis_types']:
                score += 3
                reasons.append(f"Crisis type match: {crisis_type}")

        # Crisis subtype matching
        if 'crisis_subtypes' in rules:
            scenario_subtypes = [s.lower() for s in scenario_meta.get('crisis_subtypes', [])]
            matching_subtypes = set(rules['crisis_subtypes']) & set(scenario_subtypes)
            if matching_subtypes:
                score += 2
                reasons.append(f"Subtype match: {', '.join(matching_subtypes)}")

        # Domain matching
        if 'domains' in rules:
            affected_domains = [d.lower() for d in scenario_meta.get('affected_domains', [])]
            matching_domains = set(rules['domains']) & set(affected_domains)
            if matching_domains:
                score += 2
                reasons.append(f"Domain match: {', '.join(matching_domains)}")

        # Severity threshold
        if 'severity_threshold' in rules:
            severity = scenario_meta.get('severity', 0)
            if severity >= rules['severity_threshold']:
                score += 1
                reasons.append(f"Severity {severity:.2f} >= threshold {rules['severity_threshold']}")

        # Geographic scope
        if 'geographic_scope' in rules:
            scope = scenario_meta.get('geographic_scope', '').lower()
            if scope in rules['geographic_scope']:
                score += 2
                reasons.append(f"Geographic scope: {scope}")

        # Geographic location
        if 'geographic_location' in rules:
            location = scenario_meta.get('geographic_location', '').lower()
            if location in rules['geographic_location']:
                score += 2
                reasons.append(f"Geographic location: {location}")

        # Command structure
        if 'command_structure' in rules:
            needed = scenario_meta.get('command_structure_needed', {})
            for cs in rules['command_structure']:
                if needed.get(cs):
                    score += 2
                    reasons.append(f"Command structure: {cs}")

        # Multi-jurisdictional
        if rules.get('multi_jurisdictional'):
            if scenario_meta.get('command_structure_needed', {}).get('multi_jurisdictional'):
                score += 1
                reasons.append("Multi-jurisdictional coordination needed")

        # Infrastructure systems
        if 'infrastructure_systems' in rules:
            scenario_systems = [s.lower() for s in scenario_meta.get('infrastructure_systems', [])]
            matching_systems = set(rules['infrastructure_systems']) & set(scenario_systems)
            if matching_systems:
                score += 2
                reasons.append(f"Infrastructure: {', '.join(matching_systems)}")

        # Affected population threshold
        if 'affected_populations_threshold' in rules:
            affected = scenario_meta.get('affected_populations', 0)
            if affected >= rules['affected_populations_threshold']:
                score += 1
                reasons.append(f"Affected population {affected} >= {rules['affected_populations_threshold']}")

        # Duration threshold
        if 'duration_hours_threshold' in rules:
            duration = scenario_meta.get('duration_estimated_hours', 0)
            if duration >= rules['duration_hours_threshold']:
                score += 1
                reasons.append(f"Duration {duration}h >= {rules['duration_hours_threshold']}h")

        return (score, reasons)

    def _log_selection_details(
        self,
        scenario: Dict,
        selected: List[str],
        reasons: Dict
    ) -> None:
        """Log detailed information about expert selection."""
        logger.info("=" * 70)
        logger.info("EXPERT SELECTION ANALYSIS")
        logger.info("=" * 70)

        meta = scenario.get('expert_selection', {})
        logger.info(f"Scenario: {scenario.get('title', 'Untitled')}")
        logger.info(f"Crisis Type: {meta.get('crisis_type', 'N/A')}")
        logger.info(f"Severity: {meta.get('severity', 'N/A')}")
        logger.info(f"Scope: {meta.get('geographic_scope', 'N/A')}")
        logger.info(f"Affected Population: {meta.get('affected_populations', 'N/A')}")
        logger.info("")

        logger.info(f"Selected {len(selected)} experts:")
        for agent_id in selected:
            if agent_id in reasons:
                score = reasons[agent_id]['score']
                desc = reasons[agent_id]['description']
                reason_list = reasons[agent_id]['reasons']
                logger.info(f"\n  ✓ {agent_id} (score: {score})")
                logger.info(f"    {desc}")
                for reason in reason_list:
                    logger.info(f"    - {reason}")

        logger.info("=" * 70)

    def get_expert_description(self, agent_id: str) -> str:
        """
        Get human-readable description of an expert role.

        Args:
            agent_id: Agent ID

        Returns:
            Description string
        """
        for rules in self.EXPERT_SELECTION_RULES.values():
            if rules.get('agent_id') == agent_id:
                return rules.get('description', 'No description available')
        return 'Unknown expert'

    def explain_selection(self, scenario: Dict) -> Dict[str, Any]:
        """
        Provide detailed explanation of why experts were selected.

        Args:
            scenario: Scenario dictionary

        Returns:
            Dictionary with selection explanation
        """
        selected = self.select_experts(scenario)

        explanation = {
            'scenario_summary': {
                'title': scenario.get('title', 'Untitled'),
                'crisis_type': scenario.get('expert_selection', {}).get('crisis_type'),
                'severity': scenario.get('expert_selection', {}).get('severity'),
                'scope': scenario.get('expert_selection', {}).get('geographic_scope')
            },
            'selected_experts': [],
            'total_count': len(selected)
        }

        for agent_id in selected:
            explanation['selected_experts'].append({
                'agent_id': agent_id,
                'description': self.get_expert_description(agent_id)
            })

        return explanation


# Convenience function for quick access
def select_experts_for_scenario(scenario: Dict, verbose: bool = False) -> List[str]:
    """
    Quick helper function to select experts for a scenario.

    Args:
        scenario: Scenario dictionary with expert_selection metadata
        verbose: Enable verbose logging

    Returns:
        List of agent IDs to engage

    Example:
        >>> from scenarios.expert_selector import select_experts_for_scenario
        >>> scenario = load_scenario('scenarios/flood_scenario.json')
        >>> agents = select_experts_for_scenario(scenario, verbose=True)
    """
    selector = ExpertSelector(verbose=verbose)
    return selector.select_experts(scenario)
