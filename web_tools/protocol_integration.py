"""
Protocol-to-Scenario Integration Module

PURPOSE:
Integrates Incident Handling Protocols (LLM Training Q&A) with Crisis Scenarios
(Multi-Agent Decision Making) to leverage expert knowledge for action creation.

INTEGRATION GOALS:
1. Link protocols to relevant crisis scenarios by type/category
2. Extract actionable steps from protocol answers
3. Suggest response actions based on protocol knowledge
4. Use protocol expertise to help score action criteria

DESIGN:
- Protocols contain expert Q&A for training
- Crisis scenarios contain available_actions for decision-making
- This module bridges the two by analyzing protocol answers to suggest actions
"""

import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path


class ProtocolIntegration:
    """
    Integrates Incident Handling Protocols with Crisis Scenarios.
    """

    def __init__(self, protocols_file: str = None):
        """
        Initialize the protocol integration system.

        Args:
            protocols_file: Path to protocols JSON file
        """
        if protocols_file is None:
            protocols_file = Path(__file__).parent.parent / 'web_tools' / 'data' / 'scenarios.json'

        self.protocols_file = Path(protocols_file)
        self.protocols = self._load_protocols()

    def _load_protocols(self) -> List[Dict[str, Any]]:
        """Load protocols from JSON file."""
        if not self.protocols_file.exists():
            return []

        with open(self.protocols_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_relevant_protocols(
        self,
        crisis_type: str,
        category: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get protocols relevant to a crisis type.

        Args:
            crisis_type: Type of crisis (e.g., 'wildfire', 'flood')
            category: Optional category filter (e.g., 'firefighting')
            limit: Maximum number of protocols to return

        Returns:
            List of relevant protocol dictionaries
        """
        relevant = []

        # Map crisis types to protocol categories
        type_to_category = {
            'wildfire': ['firefighting', 'disaster'],
            'flood': ['disaster', 'search_rescue'],
            'earthquake': ['disaster', 'search_rescue', 'medical'],
            'hazmat': ['firefighting', 'disaster'],
            'mass_casualty': ['medical', 'disaster'],
            'explosion': ['firefighting', 'medical', 'disaster'],
            'pandemic': ['medical', 'disaster'],
            'civil_unrest': ['police', 'disaster'],
            'terrorist': ['police', 'medical', 'disaster']
        }

        # Get relevant categories for this crisis type
        relevant_categories = type_to_category.get(crisis_type.lower(), ['general'])
        if category:
            relevant_categories = [category]

        # Filter protocols by category and tags
        for protocol in self.protocols:
            # Check category match
            if protocol.get('category') in relevant_categories:
                relevant.append(protocol)
                continue

            # Check tags for crisis type
            tags = protocol.get('tags', [])
            if crisis_type.lower() in [tag.lower() for tag in tags]:
                relevant.append(protocol)

        return relevant[:limit]

    def extract_actions_from_protocol(
        self,
        protocol: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Extract potential actions from a protocol's answer.

        This analyzes the expert's answer to identify actionable steps
        that could become response actions in a crisis scenario.

        Args:
            protocol: Protocol dictionary with 'answer' field

        Returns:
            List of extracted action suggestions
        """
        answer = protocol.get('answer', '')
        if not answer:
            return []

        actions = []

        # Look for action patterns in the answer
        # Pattern 1: Numbered lists (1., 2., etc.)
        numbered_pattern = r'(\d+)[.):]\s*([^\n]+)'
        matches = re.findall(numbered_pattern, answer)
        for num, action_text in matches:
            if len(action_text.strip()) > 20:  # Ignore very short items
                actions.append({
                    'name': action_text.strip()[:100],
                    'description': action_text.strip(),
                    'source': 'numbered_list'
                })

        # Pattern 2: Bullet points (-, *, •)
        bullet_pattern = r'[•\-\*]\s*([^\n]+)'
        matches = re.findall(bullet_pattern, answer)
        for action_text in matches:
            if len(action_text.strip()) > 20:
                actions.append({
                    'name': action_text.strip()[:100],
                    'description': action_text.strip(),
                    'source': 'bullet_point'
                })

        # Pattern 3: Action verbs at sentence start
        action_verbs = [
            'evacuate', 'deploy', 'establish', 'coordinate', 'alert',
            'mobilize', 'activate', 'implement', 'initiate', 'conduct',
            'provide', 'secure', 'contain', 'isolate', 'monitor'
        ]

        sentences = re.split(r'[.!?]', answer)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 30:  # Meaningful length
                first_word = sentence.split()[0].lower() if sentence else ''
                if first_word in action_verbs:
                    actions.append({
                        'name': sentence[:100],
                        'description': sentence,
                        'source': 'action_verb'
                    })

        # Deduplicate similar actions
        unique_actions = []
        seen_names = set()
        for action in actions:
            name_key = action['name'].lower()[:50]  # First 50 chars for similarity
            if name_key not in seen_names:
                seen_names.add(name_key)
                unique_actions.append(action)

        return unique_actions[:10]  # Limit to top 10

    def suggest_action_from_protocol(
        self,
        crisis_type: str,
        action_context: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Suggest response actions based on protocol knowledge.

        Args:
            crisis_type: Type of crisis
            action_context: Additional context for action suggestion

        Returns:
            List of suggested actions with initial scores
        """
        relevant_protocols = self.get_relevant_protocols(crisis_type, limit=3)

        all_suggestions = []

        for protocol in relevant_protocols:
            extracted_actions = self.extract_actions_from_protocol(protocol)

            for action in extracted_actions:
                # Create a suggested action with default criteria scores
                suggestion = {
                    'id': f"action_{len(all_suggestions) + 1}",
                    'name': action['name'],
                    'description': action['description'],
                    'required_resources': self._infer_resources(action['description']),
                    'estimated_duration': '2-4 hours',  # Default
                    'risk_level': 0.5,  # Default medium risk
                    'criteria_scores': self._estimate_criteria_scores(
                        action['description'],
                        crisis_type
                    ),
                    'source_protocol': protocol.get('id', 'unknown'),
                    'source_type': action.get('source', 'unknown')
                }
                all_suggestions.append(suggestion)

        return all_suggestions[:5]  # Return top 5 suggestions

    def _infer_resources(self, description: str) -> List[str]:
        """
        Infer required resources from action description.

        Args:
            description: Action description text

        Returns:
            List of inferred resource names
        """
        resources = []
        description_lower = description.lower()

        # Resource keyword mapping
        resource_keywords = {
            'fire_trucks': ['fire truck', 'engine', 'apparatus'],
            'ambulances': ['ambulance', 'ems', 'paramedic'],
            'police_units': ['police', 'law enforcement', 'officer'],
            'evacuation_buses': ['bus', 'evacuate', 'evacuation'],
            'helicopters': ['helicopter', 'aircraft', 'aerial'],
            'rescue_teams': ['rescue', 'search and rescue', 'sar'],
            'medical_supplies': ['medical', 'supplies', 'equipment'],
            'shelter': ['shelter', 'refuge', 'accommodation'],
            'personnel': ['personnel', 'staff', 'crew', 'team'],
            'heavy_equipment': ['bulldozer', 'excavator', 'crane']
        }

        for resource, keywords in resource_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                resources.append(resource)

        return resources if resources else ['general_resources']

    def _estimate_criteria_scores(
        self,
        description: str,
        crisis_type: str
    ) -> Dict[str, float]:
        """
        Estimate criteria scores based on action description.

        This provides initial estimates that can be refined by the user.

        Args:
            description: Action description
            crisis_type: Type of crisis

        Returns:
            Dictionary of criteria scores (0.0-1.0)
        """
        scores = {
            'effectiveness': 0.7,  # Default moderate effectiveness
            'safety': 0.7,
            'speed': 0.6,
            'cost': 0.5,
            'public_acceptance': 0.7
        }

        description_lower = description.lower()

        # Adjust based on keywords in description

        # Effectiveness indicators
        if any(word in description_lower for word in ['immediate', 'comprehensive', 'complete']):
            scores['effectiveness'] += 0.1

        # Safety indicators
        if any(word in description_lower for word in ['safe', 'secure', 'protect']):
            scores['safety'] += 0.15
        if any(word in description_lower for word in ['risk', 'danger', 'hazard']):
            scores['safety'] -= 0.1

        # Speed indicators
        if any(word in description_lower for word in ['immediate', 'rapid', 'quick', 'fast']):
            scores['speed'] += 0.15
        if any(word in description_lower for word in ['gradual', 'slow', 'long-term']):
            scores['speed'] -= 0.1

        # Cost indicators (higher = cheaper)
        if any(word in description_lower for word in ['expensive', 'costly', 'aircraft', 'helicopter']):
            scores['cost'] -= 0.2
        if any(word in description_lower for word in ['efficient', 'economical']):
            scores['cost'] += 0.1

        # Public acceptance indicators
        if any(word in description_lower for word in ['evacua', 'relocate', 'displace']):
            scores['public_acceptance'] -= 0.1
        if any(word in description_lower for word in ['voluntary', 'community', 'public']):
            scores['public_acceptance'] += 0.1

        # Ensure all scores are in valid range
        for key in scores:
            scores[key] = max(0.0, min(1.0, scores[key]))
            scores[key] = round(scores[key], 2)

        return scores

    def link_protocol_to_scenario(
        self,
        protocol_id: str,
        scenario_id: str
    ) -> Dict[str, str]:
        """
        Create a link between a protocol and a scenario.

        Args:
            protocol_id: ID of the protocol
            scenario_id: ID of the crisis scenario

        Returns:
            Link metadata
        """
        return {
            'protocol_id': protocol_id,
            'scenario_id': scenario_id,
            'linked_at': str(Path(__file__).stat().st_mtime),
            'link_type': 'knowledge_base'
        }


# Singleton instance
_protocol_integration = None

def get_protocol_integration() -> ProtocolIntegration:
    """Get singleton instance of ProtocolIntegration."""
    global _protocol_integration
    if _protocol_integration is None:
        _protocol_integration = ProtocolIntegration()
    return _protocol_integration
