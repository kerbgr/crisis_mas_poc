"""
Offline DQS recalculation with L1-normalized MCDA scores.

Reads every er/results.json and gat/results.json already on disk,
applies L1 normalization to the raw MCDA (TOPSIS) scores so both
components share the same [0,1] distribution scale, recomputes
combined scores, and writes:

  <run_dir>/<method>/dqs_breakdown_recalc.png   — updated breakdown plot
  <run_dir>/<method>/dqs_recalculated.json      — updated score fields

No LLM calls are made.  Original results.json files are not modified.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_ROOT = Path(__file__).parent.parent / 'results'


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def l1_normalize(scores: dict) -> dict:
    total = sum(scores.values())
    if total <= 0:
        n = len(scores)
        return {k: 1.0 / n for k in scores} if n else scores
    return {k: v / total for k, v in scores.items()}


def recalculate(decision: dict) -> dict:
    """Return updated score fields; does not mutate the input dict."""
    er_scores   = decision.get('er_scores', {})
    mcda_raw    = decision.get('mcda_scores', {})
    mcda_norm   = l1_normalize(mcda_raw)

    # Rebuild combined scores with normalised MCDA
    all_alts = set(list(er_scores.keys()) + list(mcda_raw.keys()))
    combined = {
        alt: 0.6 * er_scores.get(alt, 0.0) + 0.4 * mcda_norm.get(alt, 0.0)
        for alt in all_alts
    }

    recommended = max(combined, key=combined.__getitem__) if combined else None

    return {
        'er_scores':       er_scores,
        'mcda_scores_raw': mcda_raw,
        'mcda_scores':     mcda_norm,
        'final_scores':    combined,
        'recommended_alternative': recommended,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _short(name: str) -> str:
    return name.replace('action_', '').replace('_', ' ').title()


def plot_dqs_breakdown(updated: dict, original_rec: str, out_path: Path,
                       run_label: str, method: str) -> None:
    final    = updated['final_scores']
    er_s     = updated['er_scores']
    mcda_n   = updated['mcda_scores']
    mcda_raw = updated['mcda_scores_raw']
    rec      = updated['recommended_alternative']

    alts   = sorted(final, key=final.__getitem__, reverse=True)
    n      = len(alts)
    labels = [_short(a) for a in alts]
    y      = np.arange(n)
    h      = 0.22

    colors = sns.color_palette('Set2', 10)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, n * 0.85)),
                             gridspec_kw={'width_ratios': [2, 1]})

    # --- Left panel: ER / MCDA-norm / Combined ---
    ax = axes[0]
    b_er   = ax.barh(y + h,   [er_s.get(a, 0)   for a in alts], h, label='ER Belief',       color=colors[0], alpha=0.85)
    b_mcda = ax.barh(y,       [mcda_n.get(a, 0) for a in alts], h, label='MCDA (norm)',     color=colors[1], alpha=0.85)
    b_comb = ax.barh(y - h,   [final.get(a, 0)  for a in alts], h, label='Combined DQS',    color=colors[2], alpha=0.95)

    for i, (bar, a) in enumerate(zip(b_comb, alts)):
        is_rec = (a == rec)
        val = final[a]
        ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}{"  *" if is_rec else ""}',
                va='center', ha='left', fontsize=8,
                fontweight='bold' if is_rec else 'normal')
        if is_rec:
            for b in [b_er, b_mcda, b_comb]:
                b[i].set_edgecolor('goldenrod')
                b[i].set_linewidth(2.0)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Score', fontsize=10)
    ax.set_title(f'DQS Decomposition  (60% ER + 40% MCDA-norm)\n{run_label} [{method.upper()}]',
                 fontsize=10, fontweight='bold')
    ax.set_xlim(0, min(1.0, max(final.values()) * 1.35))
    ax.legend(fontsize=8, loc='lower right')

    # --- Right panel: raw TOPSIS vs norm MCDA ---
    ax2 = axes[1]
    raw_vals  = [mcda_raw.get(a, 0) for a in alts]
    norm_vals = [mcda_n.get(a, 0)   for a in alts]

    ax2.barh(y + h / 2, raw_vals,  h, label='TOPSIS (raw)',  color=colors[4], alpha=0.75)
    ax2.barh(y - h / 2, norm_vals, h, label='MCDA (norm)', color=colors[1], alpha=0.85)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel('MCDA score', fontsize=9)
    ax2.set_title('TOPSIS raw vs normalised', fontsize=9)
    ax2.legend(fontsize=8)

    # Footnote
    rec_changed = (rec != original_rec)
    note = ('* Recommended  |  gold border = recommended'
            + (f'  |  NOTE: recommendation changed from {_short(original_rec)}'
               if rec_changed else ''))
    fig.text(0.01, -0.03, note, fontsize=7.5, color='#555' if not rec_changed else '#c00',
             style='italic')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def process_file(json_path: Path) -> dict:
    data    = json.loads(json_path.read_text())
    decision = data.get('decision', {})

    if not decision.get('er_scores') or not decision.get('mcda_scores'):
        return {'skipped': True, 'reason': 'missing er_scores or mcda_scores'}

    updated = recalculate(decision)

    # Build run label from path
    parts     = json_path.parts
    scenario  = parts[parts.index('results') + 1]
    run_dir   = parts[parts.index('results') + 2]
    method    = parts[parts.index('results') + 3]
    run_label = f'{scenario}/{run_dir}'

    # Plot
    plot_path = json_path.parent / 'dqs_breakdown_recalc.png'
    plot_dqs_breakdown(
        updated,
        original_rec=decision.get('recommended_alternative', ''),
        out_path=plot_path,
        run_label=run_label,
        method=method
    )

    # Save recalculated scores (preserves original results.json untouched)
    out = {
        'source':          str(json_path),
        'scenario':        scenario,
        'run':             run_dir,
        'method':          method,
        'llm_provider':    data.get('llm_provider', 'unknown'),
        'timestamp':       data.get('timestamp', ''),
        'original': {
            'recommended_alternative': decision.get('recommended_alternative'),
            'final_scores':            decision.get('final_scores', {}),
            'mcda_scores_raw':         decision.get('mcda_scores', {}),
        },
        'recalculated': {
            'recommended_alternative': updated['recommended_alternative'],
            'er_scores':               updated['er_scores'],
            'mcda_scores_raw':         updated['mcda_scores_raw'],
            'mcda_scores_norm':        updated['mcda_scores'],
            'final_scores':            updated['final_scores'],
        },
        'recommendation_changed': (
            updated['recommended_alternative'] != decision.get('recommended_alternative')
        ),
    }
    out_json = json_path.parent / 'dqs_recalculated.json'
    out_json.write_text(json.dumps(out, indent=2))

    return out


def main():
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.1)
    plt.rcParams['axes.unicode_minus'] = False

    files = sorted(RESULTS_ROOT.rglob('*/results.json'))
    # Only per-method files (inside er/ or gat/ subdirs)
    files = [f for f in files if f.parent.name in ('er', 'gat')]

    print(f'Found {len(files)} result files across scenarios:')
    for s in sorted({f.parts[f.parts.index('results') + 1] for f in files}):
        n = sum(1 for f in files if f.parts[f.parts.index('results') + 1] == s)
        print(f'  {s}: {n} files')
    print()

    changed, skipped, errors = [], [], []

    for i, path in enumerate(files, 1):
        label = '/'.join(path.parts[-4:])
        try:
            result = process_file(path)
            if result.get('skipped'):
                skipped.append((label, result['reason']))
                print(f'  [{i:3d}/{len(files)}] SKIP  {label}  ({result["reason"]})')
            elif result['recommendation_changed']:
                changed.append(label)
                orig = result['original']['recommended_alternative']
                new  = result['recalculated']['recommended_alternative']
                print(f'  [{i:3d}/{len(files)}] CHANGED  {label}  {orig} -> {new}')
            else:
                print(f'  [{i:3d}/{len(files)}] OK    {label}')
        except Exception as e:
            errors.append((label, str(e)))
            print(f'  [{i:3d}/{len(files)}] ERROR {label}: {e}')

    # Summary
    print()
    print('=' * 70)
    print(f'DONE  processed={len(files) - len(skipped) - len(errors)}  '
          f'skipped={len(skipped)}  errors={len(errors)}  '
          f'recommendation_changed={len(changed)}')
    if changed:
        print()
        print('Runs where recommendation changed after normalisation:')
        for c in changed:
            print(f'  {c}')
    if errors:
        print()
        print('Errors:')
        for e_label, e_msg in errors:
            print(f'  {e_label}: {e_msg}')
    print()
    print(f'Plots saved as dqs_breakdown_recalc.png in each run/method directory.')
    print(f'Scores saved as dqs_recalculated.json in each run/method directory.')


if __name__ == '__main__':
    main()
