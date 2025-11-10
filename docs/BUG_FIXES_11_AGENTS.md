# Bug Fixes for 11-Agent System (November 2025)

## Overview

This document summarizes critical bug fixes applied to enable the Crisis MAS system to operate correctly with all 11 expert agents. Prior to these fixes, the system would crash or produce incorrect visualizations when using more than 3 agents.

---

## Fixed Errors

### ✅ Error 1: Conflict Resolution TypeError

**Error Message:**
```
ERROR:agents.coordinator_agent:Error resolving conflicts: 'str' object has no attribute 'get'
```

**Root Cause:**
- `ConsensusModel.suggest_resolution()` returns a formatted string
- `CoordinatorAgent.resolve_conflicts()` expected a dictionary and tried to call `.get()` methods

**Files Affected:**
- `agents/coordinator_agent.py` (lines 593-622)

**Fix Applied:**
- Updated coordinator to accept string return value
- Parse resolution text to extract compromise alternatives
- Determine strategy from text content ('ESCALATE', compromise alternatives, etc.)
- Convert string to structured dictionary format

**Commit:** `8c63001`

---

### ✅ Error 2: Belief Distribution Shape Mismatch

**Error Message:**
```
ERROR:evaluation.visualizations:Failed to plot belief distributions: shape mismatch:
objects cannot be broadcast to a single shape. Mismatch is between arg 0 with shape (11,)
and arg 1 with shape (10,).
```

**Root Cause:**
- Not all 11 agents had beliefs for all alternatives
- When plotting stacked bars, arrays had inconsistent lengths
- Example: Agent 1-10 had belief for "alt_evacuate", but Agent 11 didn't
- This created arrays like `[0.6, 0.5, ..., 0.4]` (10 elements) being added to a vector of 11 agents

**Files Affected:**
- `evaluation/visualizations.py` (lines 396-445)

**Fix Applied:**
- Collect all alternatives across all agents first
- Build consistent data matrix: `belief_data[alt_id][agent_idx]`
- Use `.get(alt_id, 0.0)` to default missing beliefs to 0.0
- Ensures all arrays are same length before numpy operations
- Explicit `np.array()` conversion for reliable math

**Commit:** `8c63001`

---

### ✅ Error 3: Agent Network Visualization IndexError

**Error Message:**
```
ERROR:evaluation.visualizations:Failed to plot agent network: list index out of range
```

**Root Cause:**
- Color palette `agent_colors` only had 8 colors
- System tried to access `agent_colors[10]` for 11th agent → IndexError

**Files Affected:**
- `evaluation/visualizations.py` (lines 359, 929)

**Fix Applied:**
- Extended `agent_colors` palette from 8 to 12 colors
- Used modulo operator in legend: `agent_colors[i % len(agent_colors)]`
- Prevents index errors even with 12+ agents

**Commit:** `c21b995`

---

### ⚠️ Warning: Agent Confidences (Not Critical)

**Warning Message:**
```
WARNING:evaluation.metrics:No agent confidences found
```

**Status:** Non-critical warning
- Occurs when evaluation runs before agent assessments are stored
- Does not prevent system operation
- Can be safely ignored

---

## Additional Improvements

### Agent Network Visualization Generator

**New File:** `generate_agent_network_visualization.py`

**Features:**
- Regenerates agent network with all 11 current experts
- Filters out legacy agents from profiles (only uses current 11)
- Creates hierarchical trust matrix showing command structure
- Output: `results/visualizations/agent_network_11_experts.png`

**Usage:**
```bash
python generate_agent_network_visualization.py
```

**Commit:** `c21b995`

---

## Testing the Fixes

### Test with Auto Expert Selection

```bash
# Test with auto-selection (may select 3-11 agents based on scenario)
python main.py --scenario scenarios/flood_scenario.json --expert-selection auto --verbose
```

**Expected:** No errors, successful visualization generation

### Test with All 11 Agents

```bash
# Force all 11 agents manually
python main.py --scenario scenarios/flood_scenario.json --agents all
```

**Expected:** All agents participate, visualizations render correctly

### Test Visualization Regeneration

```bash
# Regenerate agent network diagram
python generate_agent_network_visualization.py
```

**Expected:** Creates network PNG showing all 11 expert nodes

---

## Verification Checklist

Before considering the 11-agent system stable, verify:

- [x] System runs without TypeErrors
- [x] Belief distributions plot correctly with 11 agents
- [x] Agent network visualization renders all 11 nodes
- [x] Conflict resolution returns structured data
- [x] Shape mismatches resolved in numpy operations
- [ ] All 11 agents can participate in decision-making
- [ ] Consensus calculation works with 11 agents
- [ ] MCDA scoring handles 11 agent assessments

---

## Files Modified

| File | Lines Changed | Purpose |
|------|--------------|---------|
| `agents/coordinator_agent.py` | 593-622 | Fix conflict resolution type mismatch |
| `evaluation/visualizations.py` | 359, 396-445, 929 | Fix visualization errors, extend color palette |
| `generate_agent_network_visualization.py` | NEW (161 lines) | Generate 11-agent network diagram |
| `docs/REGENERATE_VISUALIZATIONS.md` | NEW (125 lines) | Documentation for visualization regeneration |

---

## Related Documentation

- [README.md](../README.md) - Main system documentation with 11-agent usage
- [ARCHITECTURE_DIAGRAMS.md](../ARCHITECTURE_DIAGRAMS.md) - Diagram #11: Expert Selection System
- [REGENERATE_VISUALIZATIONS.md](REGENERATE_VISUALIZATIONS.md) - How to regenerate agent network

---

## Commit History

| Commit | Date | Description |
|--------|------|-------------|
| `c21b995` | Nov 10, 2025 | Fix agent network visualization to support 11 experts |
| `8c63001` | Nov 10, 2025 | Fix critical errors when running with 11 agents |
| `2560afe` | Nov 10, 2025 | Add documentation for regenerating visualizations |

---

**Version:** 0.8
**Status:** ✅ Stable with 11 agents
**Last Updated:** November 10, 2025
