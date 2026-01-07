import asyncio
import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple

try:
    from .utils import *
except ImportError:
    from utils import *

# Compile regex patterns once for performance
HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$')
CODE_BLOCK_PATTERN = re.compile(r'^```')


async def get_node_summary(node: Dict[str, Any], summary_token_threshold: int = 200, model: Optional[str] = None) -> str:
    """Returns the node text directly if under token limit, otherwise generates a summary."""
    node_text = node.get('text', '')
    num_tokens = count_tokens(node_text, model=model)
    
    if num_tokens < summary_token_threshold:
        return node_text
    else:
        return await generate_node_summary(node, model=model)


async def generate_summaries_for_structure_md(
    structure: List[Dict[str, Any]], 
    summary_token_threshold: int, 
    model: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Generates summaries for all nodes in the structure concurrently."""
    nodes = structure_to_list(structure)
    
    # Create tasks for all nodes
    tasks = [
        get_node_summary(node, summary_token_threshold=summary_token_threshold, model=model) 
        for node in nodes
    ]
    summaries = await asyncio.gather(*tasks)
    
    # Assign summaries back to the nodes
    for node, summary in zip(nodes, summaries):
        if not node.get('nodes'):
            node['summary'] = summary
        else:
            node['prefix_summary'] = summary
            
    return structure


def extract_nodes_from_markdown(markdown_content: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Extracts header nodes from markdown content, skipping code blocks."""
    node_list = []
    lines = markdown_content.split('\n')
    in_code_block = False
    
    for line_num, line in enumerate(lines, 1):
        stripped_line = line.strip()
        
        # Toggle code block state
        if CODE_BLOCK_PATTERN.match(stripped_line):
            in_code_block = not in_code_block
            continue
        
        # Skip empty lines or lines inside code blocks
        if not stripped_line or in_code_block:
            continue
        
        # Check for headers
        match = HEADER_PATTERN.match(stripped_line)
        if match:
            title = match.group(2).strip()
            node_list.append({'node_title': title, 'line_num': line_num})

    return node_list, lines


def extract_node_text_content(node_list: List[Dict[str, Any]], markdown_lines: List[str]) -> List[Dict[str, Any]]:
    """Extracts the full text content for each node based on line numbers."""
    all_nodes = []
    
    # First pass: Create nodes with levels
    for node in node_list:
        line_content = markdown_lines[node['line_num'] - 1]
        header_match = HEADER_PATTERN.match(line_content)
        
        if not header_match:
            print(f"Warning: Line {node['line_num']} does not contain a valid header: '{line_content}'")
            continue
            
        processed_node = {
            'title': node['node_title'],
            'line_num': node['line_num'],
            'level': len(header_match.group(1))
        }
        all_nodes.append(processed_node)
    
    # Second pass: Extract text segments between headers
    for i, node in enumerate(all_nodes):
        start_line = node['line_num'] - 1
        # Determine end line (start of next node or end of file)
        if i + 1 < len(all_nodes):
            end_line = all_nodes[i + 1]['line_num'] - 1
        else:
            end_line = len(markdown_lines)
        
        node['text'] = '\n'.join(markdown_lines[start_line:end_line]).strip()
    
    return all_nodes


def _get_children_indices(node_list: List[Dict[str, Any]], parent_index: int) -> List[int]:
    """Helper to find all direct and indirect children indices of a parent node."""
    parent_level = node_list[parent_index]['level']
    children_indices = []
    
    for i in range(parent_index + 1, len(node_list)):
        current_level = node_list[i]['level']
        
        # Stop if we hit a node at the same or higher level (sibling or parent's sibling)
        if current_level <= parent_level:
            break
            
        children_indices.append(i)
    
    return children_indices


def update_node_list_with_text_token_count(node_list: List[Dict[str, Any]], model: Optional[str] = None) -> List[Dict[str, Any]]:
    """Updates each node with a token count that includes its own text plus all children's text."""
    # Process from end to beginning to ensure children are processed before parents
    result_list = node_list.copy()
    
    for i in range(len(result_list) - 1, -1, -1):
        current_node = result_list[i]
        children_indices = _get_children_indices(result_list, i)
        
        # Aggregate text: Self + Children
        total_text = current_node.get('text', '')
        for child_index in children_indices:
            child_text = result_list[child_index].get('text', '')
            if child_text:
                total_text += '\n' + child_text
        
        result_list[i]['text_token_count'] = count_tokens(total_text, model=model)
    
    return result_list


def tree_thinning_for_index(
    node_list: List[Dict[str, Any]], 
    min_node_token: Optional[int] = None, 
    model: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Merges small nodes into their parents if the token count is below the threshold."""
    if min_node_token is None:
        return node_list

    result_list = node_list.copy()
    nodes_to_remove = set()
    
    # Iterate backwards to handle deep leaves first
    for i in range(len(result_list) - 1, -1, -1):
        if i in nodes_to_remove:
            continue
            
        current_node = result_list[i]
        total_tokens = current_node.get('text_token_count', 0)
        
        # If node is too small, merge its children into it
        if total_tokens < min_node_token:
            children_indices = _get_children_indices(result_list, i)
            
            children_texts = []
            for child_index in sorted(children_indices):
                if child_index not in nodes_to_remove:
                    child_text = result_list[child_index].get('text', '').strip()
                    if child_text:
                        children_texts.append(child_text)
                    nodes_to_remove.add(child_index)
            
            # Perform merge
            if children_texts:
                parent_text = current_node.get('text', '')
                merged_text = parent_text
                
                # Add spacing between parent and children text
                if merged_text and not merged_text.endswith('\n'):
                    merged_text += '\n\n'
                
                merged_text += '\n\n'.join(children_texts)
                result_list[i]['text'] = merged_text
                result_list[i]['text_token_count'] = count_tokens(merged_text, model=model)
    
    # Remove merged nodes from the list
    for index in sorted(nodes_to_remove, reverse=True):
        if index < len(result_list):
            result_list.pop(index)
    
    return result_list


def build_tree_from_nodes(node_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Converts a flat list of nodes with 'level' into a hierarchical tree structure."""
    if not node_list:
        return []
    
    stack: List[Tuple[Dict[str, Any], int]] = [] # (node, level)
    root_nodes = []
    node_counter = 1
    
    for node in node_list:
        current_level = node['level']
        
        tree_node = {
            'title': node['title'],
            'node_id': str(node_counter).zfill(4),
            'text': node['text'],
            'line_num': node['line_num'],
            'nodes': []
        }
        node_counter += 1
        
        # Pop stack until we find the parent (level < current_level)
        while stack and stack[-1][1] >= current_level:
            stack.pop()
        
        if not stack:
            root_nodes.append(tree_node)
        else:
            parent_node, _ = stack[-1]
            parent_node['nodes'].append(tree_node)
        
        stack.append((tree_node, current_level))
    
    return root_nodes


def clean_tree_for_output(tree_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Recursively cleans the tree structure for final output."""
    cleaned_nodes = []
    
    for node in tree_nodes:
        cleaned_node = {
            'title': node['title'],
            'node_id': node['node_id'],
            'text': node['text'],
            'line_num': node['line_num']
        }
        
        if node['nodes']:
            cleaned_node['nodes'] = clean_tree_for_output(node['nodes'])
        
        cleaned_nodes.append(cleaned_node)
    
    return cleaned_nodes


async def md_to_tree(
    md_path: str, 
    if_thinning: bool = False, 
    min_token_threshold: Optional[int] = None, 
    if_add_node_summary: str = 'no', 
    summary_token_threshold: Optional[int] = None, 
    model: Optional[str] = None, 
    if_add_doc_description: str = 'no', 
    if_add_node_text: str = 'no', 
    if_add_node_id: str = 'yes'
) -> Dict[str, Any]:
    """Main function to convert Markdown to a structured Tree."""
    
    # Normalize boolean flags
    add_node_summary = if_add_node_summary == 'yes'
    add_doc_description = if_add_doc_description == 'yes'
    add_node_text = if_add_node_text == 'yes'
    add_node_id = if_add_node_id == 'yes'

    with open(md_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    print(f"Extracting nodes from markdown...")
    node_list, markdown_lines = extract_nodes_from_markdown(markdown_content)

    print(f"Extracting text content from nodes...")
    nodes_with_content = extract_node_text_content(node_list, markdown_lines)
    
    if if_thinning:
        nodes_with_content = update_node_list_with_text_token_count(nodes_with_content, model=model)
        print(f"Thinning nodes (Threshold: {min_token_threshold})...")
        nodes_with_content = tree_thinning_for_index(nodes_with_content, min_token_threshold, model=model)
    
    print(f"Building tree from nodes...")
    tree_structure = build_tree_from_nodes(nodes_with_content)

    if add_node_id:
        write_node_id(tree_structure)

    print(f"Formatting tree structure...")
    
    # Determine output format based on flags
    if add_node_summary:
        # Always include text initially for summary generation
        format_order = ['title', 'node_id', 'summary', 'prefix_summary', 'text', 'line_num', 'nodes']
        tree_structure = format_structure(tree_structure, order=format_order)
        
        print(f"Generating summaries for each node...")
        tree_structure = await generate_summaries_for_structure_md(
            tree_structure, 
            summary_token_threshold=summary_token_threshold or 200, 
            model=model
        )
        
        if not add_node_text:
            # Remove text after summary generation if not requested
            format_order = ['title', 'node_id', 'summary', 'prefix_summary', 'line_num', 'nodes']
            tree_structure = format_structure(tree_structure, order=format_order)
        
        if add_doc_description:
            print(f"Generating document description...")
            clean_structure = create_clean_structure_for_description(tree_structure)
            doc_description = generate_doc_description(clean_structure, model=model)
            
            return {
                'doc_name': os.path.splitext(os.path.basename(md_path))[0],
                'doc_description': doc_description,
                'structure': tree_structure,
            }
    else:
        # No summaries needed
        if add_node_text:
            format_order = ['title', 'node_id', 'summary', 'prefix_summary', 'text', 'line_num', 'nodes']
        else:
            format_order = ['title', 'node_id', 'summary', 'prefix_summary', 'line_num', 'nodes']
            
        tree_structure = format_structure(tree_structure, order=format_order)
    
    return {
        'doc_name': os.path.splitext(os.path.basename(md_path))[0],
        'structure': tree_structure,
    }


if __name__ == "__main__":
    # Configuration
    MD_NAME = 'cognitive-load'
    # Assuming this script is in a src folder and tests/results are relative to project root
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MD_PATH = os.path.join(BASE_DIR, '..', 'tests', 'markdowns', f'{MD_NAME}.md')
    OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'results')

    MODEL = "gpt-4.1"
    IF_THINNING = False
    THINNING_THRESHOLD = 5000
    SUMMARY_TOKEN_THRESHOLD = 200
    IF_SUMMARY = True

    # Execute
    tree_structure = asyncio.run(md_to_tree(
        md_path=MD_PATH, 
        if_thinning=IF_THINNING, 
        min_token_threshold=THINNING_THRESHOLD, 
        if_add_node_summary='yes' if IF_SUMMARY else 'no', 
        summary_token_threshold=SUMMARY_TOKEN_THRESHOLD, 
        model=MODEL
    ))
    
    print('\n' + '='*60)
    print('TREE STRUCTURE')
    print('='*60)
    print_json(tree_structure)

    print('\n' + '='*60)
    print('TABLE OF CONTENTS')
    print('='*60)
    print_toc(tree_structure['structure'])

    # Save Output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f'{MD_NAME}_structure.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tree_structure, f, indent=2, ensure_ascii=False)
    
    print(f"\nTree structure saved to: {output_path}")
