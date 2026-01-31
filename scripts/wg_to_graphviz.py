#!/usr/bin/env python3
"""
Generate Graphviz DOT files from workgraph state.

Usage:
    python scripts/wg_to_graphviz.py [--output PREFIX] [--filter PATTERN]

Outputs:
    - PREFIX_full.dot - Full task graph
    - PREFIX_active.dot - Only open/in-progress tasks
    - PREFIX_iteration.dot - Current iteration structure
"""

import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

def load_workgraph(path=".workgraph/graph.jsonl"):
    """Load tasks from workgraph JSONL file."""
    tasks = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("kind") == "task":
                tasks[obj["id"]] = obj
    return tasks

def status_color(status):
    """Map status to graphviz color."""
    return {
        "open": "white",
        "in-progress": "yellow",
        "done": "lightgreen",
        "failed": "lightcoral",
    }.get(status, "lightgray")

def status_shape(status):
    """Map status to graphviz shape."""
    return {
        "open": "box",
        "in-progress": "box",
        "done": "box",
        "failed": "box",
    }.get(status, "box")

def generate_dot(tasks, title="Workgraph", filter_fn=None):
    """Generate DOT representation of task graph."""
    lines = [
        f'digraph "{title}" {{',
        '  rankdir=TB;',
        '  node [fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=8];',
        '',
    ]

    # Filter tasks if needed
    if filter_fn:
        tasks = {k: v for k, v in tasks.items() if filter_fn(v)}

    # Group by iteration
    iterations = {}
    other = []
    for tid, task in tasks.items():
        title = task.get("title", "")
        if "ITERATION-" in title:
            # Extract iteration number
            import re
            match = re.search(r'ITERATION-(\d+)', title)
            if match:
                iter_num = int(match.group(1))
                if iter_num not in iterations:
                    iterations[iter_num] = []
                iterations[iter_num].append((tid, task))
        else:
            other.append((tid, task))

    # Create subgraphs for iterations
    for iter_num in sorted(iterations.keys()):
        lines.append(f'  subgraph cluster_iteration_{iter_num} {{')
        lines.append(f'    label="Iteration {iter_num}";')
        lines.append('    style=dashed;')
        lines.append('    color=blue;')
        for tid, task in iterations[iter_num]:
            status = task.get("status", "open")
            color = status_color(status)
            shape = status_shape(status)
            label = task.get("title", tid).replace(f"ITERATION-{iter_num}: ", "")
            # Escape quotes
            label = label.replace('"', '\\"')
            # Truncate long labels
            if len(label) > 40:
                label = label[:37] + "..."
            lines.append(f'    "{tid}" [label="{label}", shape={shape}, style=filled, fillcolor={color}];')
        lines.append('  }')
        lines.append('')

    # Add other tasks
    if other:
        lines.append('  subgraph cluster_other {')
        lines.append('    label="Other Tasks";')
        lines.append('    style=dashed;')
        lines.append('    color=gray;')
        for tid, task in other:
            status = task.get("status", "open")
            color = status_color(status)
            shape = status_shape(status)
            label = task.get("title", tid)
            label = label.replace('"', '\\"')
            if len(label) > 40:
                label = label[:37] + "..."
            lines.append(f'    "{tid}" [label="{label}", shape={shape}, style=filled, fillcolor={color}];')
        lines.append('  }')
        lines.append('')

    # Add edges for dependencies
    for tid, task in tasks.items():
        blocked_by = task.get("blocked_by", [])
        for dep in blocked_by:
            if dep in tasks:
                lines.append(f'  "{dep}" -> "{tid}";')

    lines.append('}')
    return '\n'.join(lines)

def generate_iteration_flow_dot(tasks):
    """Generate DOT showing the recurrent iteration pattern."""
    lines = [
        'digraph "Recurrent Sorry-Fixing Loop" {',
        '  rankdir=TB;',
        '  node [fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=9];',
        '  compound=true;',
        '',
        '  // Abstract iteration pattern',
        '  subgraph cluster_pattern {',
        '    label="Recurrent Pattern (per iteration)";',
        '    style=rounded;',
        '    color=blue;',
        '    bgcolor=aliceblue;',
        '',
        '    scan [label="1. Scan\\nsorries", shape=ellipse, style=filled, fillcolor=lightyellow];',
        '    fix [label="2. Fix sorries\\n(parallel per file)", shape=box, style=filled, fillcolor=lightblue];',
        '    build [label="3. Build\\n& test", shape=ellipse, style=filled, fillcolor=lightyellow];',
        '    verify [label="4. Verify\\ncompletion", shape=diamond, style=filled, fillcolor=lightyellow];',
        '    done [label="DONE\\n(0 sorries)", shape=doubleoctagon, style=filled, fillcolor=lightgreen];',
        '    next [label="Spawn\\nITERATION-N+1", shape=box, style=filled, fillcolor=lightcoral];',
        '',
        '    scan -> fix;',
        '    fix -> build;',
        '    build -> verify;',
        '    verify -> done [label="count=0"];',
        '    verify -> next [label="count>0"];',
        '  }',
        '',
        '  // Recurrence arrow',
        '  next -> scan [style=dashed, color=red, label="recur", constraint=false];',
        '',
        '}',
    ]
    return '\n'.join(lines)

def render_dot(dot_content, output_path, format="pdf"):
    """Render DOT to PDF/PNG using graphviz."""
    dot_path = output_path.with_suffix('.dot')
    with open(dot_path, 'w') as f:
        f.write(dot_content)

    # Try to render
    try:
        subprocess.run(
            ["dot", f"-T{format}", str(dot_path), "-o", str(output_path.with_suffix(f'.{format}'))],
            check=True,
            capture_output=True
        )
        print(f"  Generated: {output_path.with_suffix(f'.{format}')}")
    except subprocess.CalledProcessError as e:
        print(f"  Warning: Could not render {output_path} - graphviz error")
        print(f"    {e.stderr.decode() if e.stderr else 'Unknown error'}")
    except FileNotFoundError:
        print(f"  Warning: graphviz 'dot' not found - DOT file saved but not rendered")

    print(f"  DOT file: {dot_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate workgraph visualizations")
    parser.add_argument("--output", "-o", default="docs/workgraph", help="Output prefix")
    parser.add_argument("--format", "-f", default="pdf", choices=["pdf", "png", "svg"], help="Output format")
    args = parser.parse_args()

    output_prefix = Path(args.output)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    print("Loading workgraph...")
    tasks = load_workgraph()
    print(f"  Found {len(tasks)} tasks")

    # Generate full graph
    print("\nGenerating full task graph...")
    full_dot = generate_dot(tasks, title="Full Workgraph")
    render_dot(full_dot, output_prefix.with_name(f"{output_prefix.name}_full"), args.format)

    # Generate active-only graph
    print("\nGenerating active tasks graph...")
    active_dot = generate_dot(
        tasks,
        title="Active Tasks",
        filter_fn=lambda t: t.get("status") in ["open", "in-progress"]
    )
    render_dot(active_dot, output_prefix.with_name(f"{output_prefix.name}_active"), args.format)

    # Generate iteration pattern diagram
    print("\nGenerating iteration pattern diagram...")
    pattern_dot = generate_iteration_flow_dot(tasks)
    render_dot(pattern_dot, output_prefix.with_name(f"{output_prefix.name}_pattern"), args.format)

    print("\nDone!")

if __name__ == "__main__":
    main()
