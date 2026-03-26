# src/evaluation/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_loss(train_losses, save_path="figures/loss_curve.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss", color="#2ecc71", lw=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Checkpoint (every 100-200 batches)")
    plt.ylabel("Loss")
    plt.title("Transformer Training Progress")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_attention(attention, src_tokens, tgt_tokens, head_idx=0, save_path="attention_map.png"):
    """
    attention: tensor of shape [batch, heads, tgt_len, src_len]
    """
    # Remove batch dimension and pick a head
    # Shape becomes [tgt_len, src_len]
    attn = attention[0, head_idx].cpu().detach().numpy()

    # Slice the attention to match the actual token lengths
    attn = attn[:len(tgt_tokens), :len(src_tokens)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, 
                xticklabels=src_tokens, 
                yticklabels=tgt_tokens, 
                annot=False, 
                cmap='rocket', 
                cbar_kws={'label': 'Attention Score'})

    plt.xlabel("Source (English)")
    plt.ylabel("Predicted (German)")
    plt.title(f"Cross-Attention Heatmap (Head {head_idx})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f" Attention map saved to {save_path}")
    plt.show()


def export_attention_to_csv(attention, src_toks, tgt_toks, head_idx, filename="attn_matrix.csv"):
    # attention shape: [1, heads, tgt_len, src_len]
    # 1. Get the raw numpy matrix
    attn_data = attention[0, head_idx].cpu().detach().numpy()
    
    # 2. Slice both labels and data to the smallest common size 
    # to avoid the "Shape mismatch" error
    min_tgt = min(attn_data.shape[0], len(tgt_toks))
    min_src = min(attn_data.shape[1], len(src_toks))
    
    final_data = attn_data[:min_tgt, :min_src]
    final_tgt_labels = tgt_toks[:min_tgt]
    final_src_labels = src_toks[:min_src]

    # 3. Create DataFrame
    df = pd.DataFrame(final_data, index=final_tgt_labels, columns=final_src_labels)
    
    # Save and return
    df.to_csv(filename)
    print(f" Matrix exported to {filename}")
    return df.to_string()

def export_attention_to_text(attention, src_toks, tgt_toks, head_idx, save_path):
    """Simple text version for AI copy-pasting"""
    matrix_str = export_attention_to_csv(attention, src_toks, tgt_toks, head_idx, filename=save_path.replace(".txt", ".csv"))
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(matrix_str)



def generate_interactive_html(attention, src_toks, tgt_toks, filename="attention_viz.html"):
    """Creates a standalone HTML file with a head selector and SVG connections."""
    import json
    
    # Prepare data for JS: [heads, tgt_len, src_len]
    attn_data = attention[0, :, :len(tgt_toks), :len(src_toks)].cpu().detach().tolist()
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Transformer Attention Explorer</title>
        <style>
            body {{ font-family: sans-serif; display: flex; flex-direction: column; align-items: center; background: #f4f4f9; }}
            .container {{ display: flex; gap: 200px; margin-top: 50px; position: relative; }}
            .column {{ display: flex; flex-direction: column; gap: 10px; z-index: 2; }}
            .word {{ padding: 5px 10px; background: white; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; }}
            svg {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; pointer-events: none; }}
            .controls {{ margin-top: 20px; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <div class="controls">
            <label>Select Attention Head: </label>
            <input type="range" id="headSlider" min="0" max="{len(attn_data)-1}" value="0" oninput="updateViz()">
            <span id="headVal">0</span>
        </div>

        <div class="container" id="vizContainer">
            <div class="column" id="srcCol"></div>
            <div class="column" id="tgtCol"></div>
            <svg id="svgLines"></svg>
        </div>

        <script>
            const data = {json.dumps(attn_data)};
            const srcToks = {json.dumps(src_toks)};
            const tgtToks = {json.dumps(tgt_toks)};

            function init() {{
                const srcCol = document.getElementById('srcCol');
                const tgtCol = document.getElementById('tgtCol');
                srcToks.forEach(t => srcCol.innerHTML += `<div class="word">${{t}}</div>`);
                tgtToks.forEach(t => tgtCol.innerHTML += `<div class="word">${{t}}</div>`);
                setTimeout(updateViz, 100);
            }}

            function updateViz() {{
                const head = document.getElementById('headSlider').value;
                document.getElementById('headVal').innerText = head;
                const svg = document.getElementById('svgLines');
                svg.innerHTML = '';
                
                const srcWords = document.getElementById('srcCol').children;
                const tgtWords = document.getElementById('tgtCol').children;
                const weights = data[head];

                for(let t=0; t < tgtWords.length; t++) {{
                    for(let s=0; s < srcWords.length; s++) {{
                        const opacity = weights[t][s];
                        if (opacity < 0.05) continue; // Filter weak lines

                        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                        const sRect = srcWords[s].getBoundingClientRect();
                        const tRect = tgtWords[t].getBoundingClientRect();
                        const cRect = document.getElementById('vizContainer').getBoundingClientRect();

                        line.setAttribute("x1", sRect.right - cRect.left);
                        line.setAttribute("y1", sRect.top + sRect.height/2 - cRect.top);
                        line.setAttribute("x2", tRect.left - cRect.left);
                        line.setAttribute("y2", tRect.top + tRect.height/2 - cRect.top);
                        line.setAttribute("stroke", `rgba(65, 105, 225, ${{opacity * 1.5}})`);
                        line.setAttribute("stroke-width", opacity * 5);
                        svg.appendChild(line);
                    }}
                }}
            }}
            window.onload = init;
        </script>
    </body>
    </html>
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f" Interactive visualization saved to {filename}")
