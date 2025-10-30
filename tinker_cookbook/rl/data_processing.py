"""
Data processing functions for RL training.

Contains functions for computing advantages, converting trajectories to training data,
and assembling training batches.
"""

import logging
from pathlib import Path
from typing import List

import dash
import dash_cytoscape as cytoscape
from dash import html, Input, Output
import tinker
import torch
from tinker import TensorData
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup
from tinker_cookbook.utils.misc_utils import all_same, safezip
import numpy as np

logger = logging.getLogger(__name__)


def compute_advantages(trajectory_groups_P: List[TrajectoryGroup]) -> List[torch.Tensor]:
    """Compute advantages for each trajectory, centered within groups."""
    advantages_P: list[torch.Tensor] = []

    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards())
        # Center advantages within the group
        advantages_G = rewards_G - rewards_G.mean()
        advantages_P.append(advantages_G)

    return advantages_P


FlatObElem = int | tinker.ModelInputChunk
FlatOb = list[FlatObElem]


def _is_prefix(seq1: FlatOb, seq2: FlatOb) -> bool:
    """
    Check if seq1 is a prefix of seq2.
    """
    return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1


def _flat_ob_token_len(flat_ob: FlatOb) -> int:
    out = 0
    for elem in flat_ob:
        if isinstance(elem, int):
            out += 1
        else:
            out += elem.length
    return out


def _to_input_targets(model_input: tinker.ModelInput) -> tuple[tinker.ModelInput, list[int]]:
    # TODO: make this work with multimodal data
    all_ints = model_input.to_ints()
    return tinker.ModelInput.from_ints(tokens=all_ints[:-1]), all_ints[1:]


def _flat_ob_to_model_input(flat_ob: FlatOb) -> tinker.ModelInput:
    out: list[tinker.ModelInputChunk] = []
    current_text_chunk: list[int] = []

    def flush_text_chunk():
        if current_text_chunk:
            out.append(tinker.EncodedTextChunk(tokens=current_text_chunk))
            current_text_chunk.clear()

    for elem in flat_ob:
        if isinstance(elem, int):
            current_text_chunk.append(elem)
        else:
            flush_text_chunk()
            out.append(elem)
    flush_text_chunk()
    return tinker.ModelInput(chunks=out)


def _flatten_chunks(chunks: list[tinker.ModelInputChunk]) -> FlatOb:
    out: FlatOb = []
    for chunk in chunks:
        if isinstance(chunk, tinker.EncodedTextChunk):
            out.extend(chunk.tokens)
        else:
            out.append(chunk)
    return out


def trajectory_to_data(traj: Trajectory, traj_advantage: float) -> list[tinker.Datum]:
    """
    Return one or more Datum objects corresponding to the trajectory.
    If the sequence grows by appending, i.e., each successive observation contains
    the previous observation+action as a prefix, then we can return a single Datum.
    However, if we get a sequence that's not an extension of the previous sequence,
    then that results in a new Datum.

    For example, let O1 denote a chunk of observation tokens, and let A1 denote an action.

    Then let's say ob_ac_pairs is as follows.

    (O1, A1)
    (O1+A1+O2, A2)
    (O3, A3)

    Then we will merge the first two observation-action pairs into a single Datum,
    and the last observation-action pair into a separate Datum.
    """

    class SequenceAccumulator:
        full_sequence: list[FlatObElem] = []
        sampled_logprobs: list[float] = []
        advantages: list[float] = []
        mask: list[float] = []

        @classmethod
        def clear(cls):
            cls.full_sequence = []
            cls.sampled_logprobs = []
            cls.advantages = []
            cls.mask = []

    def make_datum_from_state():
        # TODO: generalize to multimodal
        all_tokens_T = _flat_ob_to_model_input(SequenceAccumulator.full_sequence)
        input_tokens_T, target_tokens_T = _to_input_targets(all_tokens_T)
        sampled_logprobs_T = SequenceAccumulator.sampled_logprobs[1:]
        advantages_T = SequenceAccumulator.advantages[1:]
        mask_T = SequenceAccumulator.mask[1:]
        assert (
            input_tokens_T.length
            == len(target_tokens_T)
            == len(sampled_logprobs_T)
            == len(advantages_T)
            == len(mask_T)
        )
        return tinker.Datum(
            model_input=input_tokens_T,
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens_T)),
                "logprobs": TensorData.from_torch(torch.tensor(sampled_logprobs_T)),
                "advantages": TensorData.from_torch(torch.tensor(advantages_T)),
                "mask": TensorData.from_torch(torch.tensor(mask_T)),
            },
        )

    data: list[tinker.Datum] = []
    for transition in traj.transitions:
        ob = transition.ob
        ob_flat = _flatten_chunks(ob.chunks)
        ac_with_logprobs = transition.ac
        if len(SequenceAccumulator.full_sequence) == 0:
            delta_ob_flat = ob_flat
        elif _is_prefix(SequenceAccumulator.full_sequence, ob_flat):
            delta_ob_flat = ob_flat[len(SequenceAccumulator.full_sequence) :]
        else:
            data.append(make_datum_from_state())
            SequenceAccumulator.clear()
            delta_ob_flat = ob_flat
        delta_ob_len = _flat_ob_token_len(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(ac_with_logprobs.tokens)
        SequenceAccumulator.sampled_logprobs.extend(
            [0.0] * delta_ob_len + ac_with_logprobs.logprobs
        )

        # Use per-token advantages if available, otherwise broadcast scalar advantage
        if transition.advantages is not None:
            # Per-token advantages computed (e.g., from branched GRPO)
            assert len(transition.advantages) == len(ac_with_logprobs.tokens), \
                f"Advantages length {len(transition.advantages)} != action tokens {len(ac_with_logprobs.tokens)}"
            action_advantages = transition.advantages
        else:
            # Traditional scalar advantage (broadcast to all action tokens)
            action_advantages = [traj_advantage] * len(ac_with_logprobs.tokens)

        SequenceAccumulator.advantages.extend(
            [0] * delta_ob_len + action_advantages
        )
        SequenceAccumulator.mask.extend([0.0] * delta_ob_len + [1.0] * len(ac_with_logprobs.tokens))

    if SequenceAccumulator.full_sequence:
        data.append(make_datum_from_state())

    return data


def assemble_training_data(
    trajectory_groups_P: List[TrajectoryGroup],
    advantages_P: List[torch.Tensor],
) -> tuple[List[tinker.Datum], List[dict[str, int]]]:
    """Convert trajectories to training data format."""
    data_D: list[tinker.Datum] = []
    metadata_D: list[dict[str, int]] = []

    for i_group, (traj_group, advantages_G) in enumerate(
        safezip(trajectory_groups_P, advantages_P)
    ):
        for i_traj, (traj, traj_advantage) in enumerate(
            safezip(traj_group.trajectories_G, advantages_G)
        ):
            # Build the full sequence from the trajectory
            new_data = trajectory_to_data(traj, float(traj_advantage))
            data_D.extend(new_data)
            metadata_D.extend([dict(group_idx=i_group, traj_idx=i_traj) for _ in new_data])

    return data_D, metadata_D


class TrieNode:
    """
    A node in a prefix trie for tracking which trajectories share token prefixes.

    Each node represents a position in the token sequence and tracks which
    trajectories pass through that position.
    """

    def __init__(self):
        self.children: dict[int, 'TrieNode'] = {}  # token_id -> child node
        self.trajectory_ids: set[int] = set()  # trajectories passing through this node


def build_prefix_trie(group: TrajectoryGroup) -> TrieNode:
    """
    Build a prefix trie from all trajectories in a group.

    For each trajectory, we create a flattened token sequence by concatenating
    observation tokens and action tokens from all transitions. We extract only
    the DELTA (new) observation tokens at each transition to avoid double-counting,
    since observations are cumulative in search environments.

    Args:
        group: A TrajectoryGroup containing trajectories to process

    Returns:
        The root node of the constructed trie
    """
    root = TrieNode()

    for traj_idx, traj in enumerate(group.trajectories_G):
        # Build flattened token sequence: [trans0.delta_ob + trans0.ac, trans1.delta_ob + trans1.ac, ...]
        current_node = root
        current_node.trajectory_ids.add(traj_idx)

        # Track cumulative token sequence to extract deltas
        full_sequence: FlatOb = []

        for transition in traj.transitions:
            # Flatten observation to handle multimodal chunks
            ob = transition.ob
            ob_flat = _flatten_chunks(ob.chunks)

            # Extract delta observation tokens (new tokens not in full_sequence)
            if len(full_sequence) == 0:
                # First transition: all tokens are new
                delta_ob_flat = ob_flat
            elif _is_prefix(full_sequence, ob_flat):
                # Observation extends previous sequence: extract only new tokens
                delta_ob_flat = ob_flat[len(full_sequence):]
            else:
                # Observation is not an extension: treat all as new
                # (This shouldn't happen in search environments but handle it)
                delta_ob_flat = ob_flat
                full_sequence = []

            # Add delta observation tokens to trie
            for elem in delta_ob_flat:
                token = elem if isinstance(elem, int) else elem
                if token not in current_node.children:
                    current_node.children[token] = TrieNode()
                current_node = current_node.children[token]
                current_node.trajectory_ids.add(traj_idx)

            # Update full_sequence with delta observation
            full_sequence.extend(delta_ob_flat)

            # Add action tokens to trie
            for token in transition.ac.tokens:
                if token not in current_node.children:
                    current_node.children[token] = TrieNode()
                current_node = current_node.children[token]
                current_node.trajectory_ids.add(traj_idx)

            # Update full_sequence with action tokens
            full_sequence.extend(transition.ac.tokens)

    return root


def compute_per_token_advantages_branched(group: TrajectoryGroup) -> None:
    """
    Compute per-token advantages for all trajectories in a branched group.

    This function implements the key innovation for branched GRPO:
    - For each token position, identify which trajectories share that prefix
    - Compute advantage using the reward statistics from only those sharing trajectories
    - Formula: advantage = (traj_reward - mean(sharing_rewards)) / std(sharing_rewards)
    - If std=0 (single trajectory at that node), set advantage=0

    The computed advantages are stored in-place in each Transition.advantages field.
    Only action tokens get advantages (observation tokens are masked during training).

    Args:
        group: A TrajectoryGroup whose trajectories should have advantages computed

    Side effects:
        Mutates each transition by setting transition.advantages for action tokens
    """
    # Build the prefix trie
    root = build_prefix_trie(group)

    # Get total rewards for all trajectories
    total_rewards = group.get_total_rewards()

    mean_total_reward = np.mean(total_rewards)
    std_total_reward = np.std(total_rewards)

    # Process each trajectory
    for traj_idx, traj in enumerate(group.trajectories_G):
        traj_reward = total_rewards[traj_idx]

        # Walk through the trie following this trajectory's token sequence
        current_node = root

        # Track cumulative token sequence to extract deltas (must match build_prefix_trie)
        full_sequence: FlatOb = []

        for transition in traj.transitions:
            # Track advantages for this transition's action tokens
            action_advantages = []

            # Flatten observation to handle multimodal chunks
            ob = transition.ob
            ob_flat = _flatten_chunks(ob.chunks)

            # Extract delta observation tokens (new tokens not in full_sequence)
            if len(full_sequence) == 0:
                # First transition: all tokens are new
                delta_ob_flat = ob_flat
            elif _is_prefix(full_sequence, ob_flat):
                # Observation extends previous sequence: extract only new tokens
                delta_ob_flat = ob_flat[len(full_sequence):]
            else:
                # Observation is not an extension: treat all as new
                delta_ob_flat = ob_flat
                full_sequence = []

            # Process delta observation tokens (walk trie but don't compute advantages)
            for elem in delta_ob_flat:
                token = elem if isinstance(elem, int) else elem
                current_node = current_node.children[token]

            # Update full_sequence with delta observation
            full_sequence.extend(delta_ob_flat)

            # Process action tokens (compute advantages)
            for token in transition.ac.tokens:
                current_node = current_node.children[token]

                # Get trajectories sharing this prefix
                sharing_traj_ids = current_node.trajectory_ids
                sharing_rewards = [total_rewards[tid] for tid in sharing_traj_ids]

                # Compute advantage
                if len(sharing_rewards) == 1:
                    # Only this trajectory has this token -> std=0 -> advantage=0
                    if std_total_reward < 1e-8:  # Numerical stability
                        advantage = 0.0
                    else:
                        advantage =(traj_reward - mean_total_reward)/std_total_reward
                else:
                    # Compute standardized advantage
                    mean_reward = np.mean(sharing_rewards)
                    std_reward = np.std(sharing_rewards)

                    if std_reward < 1e-8:  # Numerical stability
                        advantage = 0.0
                    else:
                        advantage = (traj_reward - mean_reward) / std_reward

                action_advantages.append(advantage)

            # Update full_sequence with action tokens
            full_sequence.extend(transition.ac.tokens)

            # Store advantages in the transition
            transition.advantages = action_advantages


def reconstruct_messages_at_node(
    traj: Trajectory,
    cumulative_tokens: int,
    tokenizer,
) -> list[dict]:
    """
    Reconstruct the conversation messages up to a given token position.

    This properly handles cumulative observations by extracting delta (new) tokens
    at each transition to show the full conversation flow including tool results.

    Args:
        traj: Trajectory to reconstruct from
        cumulative_tokens: Total tokens from root (includes obs + action tokens)
        tokenizer: For decoding tokens

    Returns:
        List of message dicts with 'role' and 'content'
    """
    messages = []
    total_tokens_seen = 0
    full_sequence: FlatOb = []  # Track cumulative sequence to extract deltas

    for trans_idx, transition in enumerate(traj.transitions):
        # Flatten observation to handle multimodal chunks
        ob = transition.ob
        ob_flat = _flatten_chunks(ob.chunks)
        ac_tokens = transition.ac.tokens

        if total_tokens_seen >= cumulative_tokens:
            break

        # Extract delta observation tokens (new content not in full_sequence)
        if len(full_sequence) == 0:
            # First transition: all observation tokens are new
            delta_ob_flat = ob_flat
        elif _is_prefix(full_sequence, ob_flat):
            # Observation extends previous sequence: extract only new tokens
            delta_ob_flat = ob_flat[len(full_sequence):]
        else:
            # Observation is not an extension (shouldn't happen in search envs)
            delta_ob_flat = ob_flat
            full_sequence = []

        # Decode delta observation tokens
        delta_ob_tokens = [elem if isinstance(elem, int) else elem for elem in delta_ob_flat]

        if delta_ob_tokens:
            delta_ob_text = tokenizer.decode(delta_ob_tokens)

            if trans_idx == 0:
                # First observation is system + user
                messages.append({"role": "system/user", "content": delta_ob_text})
            else:
                # Subsequent deltas are tool results / environment responses
                messages.append({"role": "tool/environment", "content": delta_ob_text})

        # Update full_sequence with delta observation
        full_sequence.extend(delta_ob_flat)
        delta_ob_len = len(delta_ob_tokens)

        # Determine how many action tokens to include
        tokens_used_so_far = total_tokens_seen + delta_ob_len
        tokens_remaining = cumulative_tokens - tokens_used_so_far

        if tokens_remaining > 0:
            # Include partial or full assistant response
            action_tokens_to_include = min(len(ac_tokens), tokens_remaining)
            partial_action_tokens = ac_tokens[:action_tokens_to_include]
            action_text = tokenizer.decode(partial_action_tokens)

            is_partial = action_tokens_to_include < len(ac_tokens)
            role = "assistant (partial)" if is_partial else "assistant"
            messages.append({"role": role, "content": action_text})

            # Update full_sequence with included action tokens
            full_sequence.extend(partial_action_tokens)

        # Update total tokens seen
        total_tokens_seen += delta_ob_len + min(len(ac_tokens), max(0, tokens_remaining))

    return messages


def compress_trie_for_visualization(
    node: "TrieNode",
    tokenizer,
    node_id: int = 0,
    parent_tokens: list[int] | None = None,
    cumulative_tokens_from_root: int = 0,
) -> tuple[dict, list[tuple[int, int]], int]:
    """
    Compress the trie by merging linear chains of nodes.

    A chain is compressed if:
    - Each node has exactly 1 child (out-degree = 1)
    - All nodes in the chain have the same trajectory_ids

    Returns:
        nodes: dict mapping node_id -> {
            'tokens': list[int],
            'text': str,
            'trajectory_ids': list[int],
            'cumulative_tokens': int
        }
        edges: list of (parent_id, child_id)
        next_node_id: Next available node ID
    """
    if parent_tokens is None:
        parent_tokens = []

    nodes = {}
    edges = []

    # Compress this chain
    current_tokens = parent_tokens.copy()
    current_node = node
    start_traj_ids = node.trajectory_ids.copy()

    # Keep compressing while conditions hold
    while (
        len(current_node.children) == 1
        and current_node.trajectory_ids == start_traj_ids
    ):
        # Get the single child
        token, child_node = next(iter(current_node.children.items()))
        current_tokens.append(token)

        # Check if child also meets compression criteria
        if (
            len(child_node.children) == 1
            and child_node.trajectory_ids == start_traj_ids
        ):
            # Continue compressing
            current_node = child_node
        else:
            # Stop here - next node branches or has different trajs
            current_node = child_node
            break

    # Calculate cumulative tokens for this node
    cumulative_tokens = cumulative_tokens_from_root + len(current_tokens)

    # Create compressed node
    if current_tokens:
        # Decode tokens to text (limit to 30 chars each for readability)
        token_texts = [tokenizer.decode([tok])[:30].replace('\n', '\\n') for tok in current_tokens]
        compressed_text = " → ".join(token_texts)
    else:
        compressed_text = "[ROOT]"

    nodes[node_id] = {
        "tokens": current_tokens,
        "text": compressed_text,
        "trajectory_ids": sorted(start_traj_ids),
        "cumulative_tokens": cumulative_tokens,
    }

    # Process children
    child_node_id = node_id + 1

    for token, child in current_node.children.items():
        # Recurse on child
        child_nodes, child_edges, next_id = compress_trie_for_visualization(
            child, tokenizer, child_node_id, [token], cumulative_tokens
        )

        # Add edge from this node to child
        edges.append((node_id, child_node_id))

        # Merge results
        nodes.update(child_nodes)
        edges.extend(child_edges)

        # Update node ID counter
        child_node_id = next_id

    return nodes, edges, child_node_id


def visualize_prefix_trie(
    group: TrajectoryGroup,
    tokenizer,
    output_path: str,
):
    """
    Create interactive HTML visualization of prefix trie using Dash-Cytoscape.

    Args:
        group: TrajectoryGroup with computed per-token advantages
        tokenizer: Tokenizer for converting token IDs to text
        output_path: Path to save the interactive HTML file
    """
    # Build the trie
    root = build_prefix_trie(group)

    # Compress for visualization
    nodes, edges, _ = compress_trie_for_visualization(root, tokenizer)

    # Convert to Cytoscape format
    cytoscape_elements = []

    # Add nodes
    for node_id, node_data in nodes.items():
        text = node_data["text"]
        traj_ids = node_data["trajectory_ids"]
        cumulative_tokens = node_data["cumulative_tokens"]
        num_trajs = len(traj_ids)

        # Create label
        traj_str = str(traj_ids)
        label = f"{text}\n{traj_str}"

        cytoscape_elements.append({
            'data': {
                'id': str(node_id),
                'label': label,
                'num_trajs': num_trajs,
                'trajectory_ids': traj_ids,
                'cumulative_tokens': cumulative_tokens,
                'text': text,
            },
            'classes': f'traj-{num_trajs}'
        })

    # Add edges
    for parent, child in edges:
        cytoscape_elements.append({
            'data': {
                'source': str(parent),
                'target': str(child),
            }
        })

    # Create Dash app
    app = dash.Dash(__name__)

    # Color mapping for different trajectory counts
    max_trajs = max(len(n["trajectory_ids"]) for n in nodes.values()) if nodes else 1
    colors = ['#e3f2fd', '#90caf9', '#42a5f5', '#1976d2', '#0d47a1']

    # Cytoscape stylesheet
    stylesheet = [
        {
            'selector': 'node',
            'style': {
                'background-color': '#90caf9',
                'width': 30,
                'height': 30,
            }
        },
        {
            'selector': 'edge',
            'style': {
                'curve-style': 'bezier',
                'target-arrow-shape': 'triangle',
                'arrow-scale': 1.5,
                'line-color': '#bbb',
                'target-arrow-color': '#bbb',
                'width': 2,
            }
        },
    ]

    # Add color styles for different trajectory counts
    for i in range(1, max_trajs + 1):
        color_idx = min(i - 1, len(colors) - 1)
        stylesheet.append({
            'selector': f'.traj-{i}',
            'style': {
                'background-color': colors[color_idx],
                'width': 50 + i * 15,
                'height': 50 + i * 15,
            }
        })

    app.layout = html.Div([
        html.H2(
            f"Prefix Trie: {len(group.trajectories_G)} trajectories, "
            f"rewards: {[f'{r:.2f}' for r in group.get_total_rewards()]}",
            style={'textAlign': 'center', 'padding': '20px'}
        ),
        html.Div([
            html.Div([
                cytoscape.Cytoscape(
                    id='cytoscape-graph',
                    elements=cytoscape_elements,
                    layout={'name': 'breadthfirst', 'directed': True, 'spacingFactor': 1.5},  # Tree layout
                    style={'width': '100%', 'height': '80vh'},
                    stylesheet=stylesheet,
                )
            ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            html.Div([
                html.H3("Click a node to view messages", style={'padding': '10px'}),
                html.Div(id='message-pane', style={'padding': '10px', 'overflowY': 'scroll', 'height': '75vh'})
            ], style={'width': '38%', 'display': 'inline-block', 'verticalAlign': 'top', 'border-left': '2px solid #ddd', 'padding-left': '10px'})
        ])
    ])

    # Callback for displaying messages
    @app.callback(
        Output('message-pane', 'children'),
        Input('cytoscape-graph', 'tapNodeData')
    )
    def display_messages(data):
        if data is None:
            return html.Div("Click a node to see message history at that point", style={'color': '#666'})

        traj_ids = data['trajectory_ids']
        cumulative_tokens = data['cumulative_tokens']
        text = data['text']

        # Get messages from first trajectory in the list
        if traj_ids and len(group.trajectories_G) > traj_ids[0]:
            traj = group.trajectories_G[traj_ids[0]]
            messages = reconstruct_messages_at_node(traj, cumulative_tokens, tokenizer)

            # Format messages for display
            message_divs = [
                html.Div([
                    html.H4(f"Trajectories: {traj_ids}", style={'color': '#1976d2'}),
                    html.P(f"Cumulative tokens: {cumulative_tokens}"),
                    html.P(f"Node text: {text}", style={'fontStyle': 'italic', 'color': '#666'}),
                    html.Hr(),
                ])
            ]

            for msg in messages:
                role = msg['role']
                content = msg['content']  # Show full content without truncation

                role_style = {
                    'system/user': {'backgroundColor': '#f5f5f5', 'padding': '10px', 'margin': '10px 0', 'borderLeft': '4px solid #2196f3'},
                    'assistant': {'backgroundColor': '#e8f5e9', 'padding': '10px', 'margin': '10px 0', 'borderLeft': '4px solid #4caf50'},
                    'assistant (partial)': {'backgroundColor': '#fff3e0', 'padding': '10px', 'margin': '10px 0', 'borderLeft': '4px solid #ff9800'},
                    'tool/environment': {'backgroundColor': '#fce4ec', 'padding': '10px', 'margin': '10px 0', 'borderLeft': '4px solid #e91e63'},
                }

                message_divs.append(
                    html.Div([
                        html.Strong(f"[{role.upper()}]"),
                        html.Pre(content, style={'whiteSpace': 'pre-wrap', 'fontSize': '12px', 'marginTop': '5px'})
                    ], style=role_style.get(role, {}))
                )

            return message_divs
        else:
            return html.Div("No messages available", style={'color': '#999'})

    # Save as standalone HTML
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate the HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Prefix Trie Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/dash-cytoscape@1.0.0/dash_cytoscape/dash_cytoscape.min.js"></script>
</head>
<body>
    <div id="react-entry-point">
        <div class="_dash-loading">
            Loading...
        </div>
    </div>
</body>
</html>
"""

    # For now, just save a note that the user needs to run the Dash app
    with open(output_path, 'w') as f:
        f.write(html_content)
        f.write("\n<!-- To view this visualization, run the Dash app in Python -->")

    logger.info(f"Prefix trie visualization saved to: {output_path}")
    print(f"\n📊 Interactive prefix trie HTML template saved to: {output_path}")
    print(f"   To view: Run `python -c 'from tinker_cookbook.rl.data_processing import run_trie_server; run_trie_server()'`")

    # Store the app for later use
    global _trie_app, _trie_port
    _trie_app = app
    _trie_port = 8050

    return app


# Global storage for the app
_trie_app = None
_trie_port = 8050


def run_trie_server(port=8050):
    """Run the Dash server for the trie visualization."""
    if _trie_app is not None:
        print(f"Starting Dash server on http://localhost:{port}")
        # Use debug=False to prevent reloader from running the script twice
        _trie_app.run(debug=False, port=port)
    else:
        print("No trie app available. Generate visualization first.")


def remove_constant_reward_groups(
    trajectory_groups_P: List[TrajectoryGroup],
) -> List[TrajectoryGroup]:
    new_groups: list[TrajectoryGroup] = []
    for group in trajectory_groups_P:
        if not all_same(group.get_total_rewards()):
            new_groups.append(group)
    if not new_groups:
        logger.warning("All rewards are uniform. There will be no gradient")
        return trajectory_groups_P[0:1]  # return singleton list in case empty
        # list will cause problems
    return new_groups
