import os
import pandas as pd
import regex
import matplotlib.pyplot as plt
import os

def plot_expertise(df):
    # for each layer depth, for each layer type, extract the max ap unit - store the results and plot them
    max_ap_per_layer_type = {}
    for layer_depth, group in df.groupby('layer_depth'):
        max_ap_per_layer_type[layer_depth] = {}
        for layer_type, type_group in group.groupby('layer_type'):
            max_ap = type_group['ap'].max()
            max_ap_per_layer_type[layer_depth][layer_type] = float(max_ap)

    # plot the results in 'max_ap_per_layer_type' dictionary
    layer_types = df.layer_type.unique().tolist()
    layer_types = ['mlp.up_proj', 'mlp.down_proj', 'mlp.gate_proj',] # 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj']

    fig, ax = plt.subplots(figsize=(15, 10), layout='tight')
    ax.set_ylim(0.60, 1.0)
    for layer_type in layer_types:
        depths = []
        aps = []
        for layer_depth in sorted(max_ap_per_layer_type.keys()):
            if layer_type in max_ap_per_layer_type[layer_depth]:
                depths.append(layer_depth)
                aps.append(max_ap_per_layer_type[layer_depth][layer_type])
        ax.plot(depths, aps, marker='o', label=layer_type)

    ax.grid(True)
    ax.set_xlabel('Layer Depth')
    ax.set_ylabel('Max AP')
    ax.legend()

    return fig, ax


def main(args):
    base_out_dir = "debug_plots"
    files = [os.path.join(args.results_dir, f, "expertise", "expertise.csv") for f in os.listdir(args.results_dir)]
    files = [f for f in files if os.path.isfile(f)]

    for f in files:
        df = pd.read_csv(f)
        
        # extact layer depth 
        df['layer_depth'] = df['layer'].str.extract(r'\.(\d+)\.').astype(int)
        df['layer_type'] = df["layer"].str.extract(r"model\.layers\.\d+\.([^:]+):0")

        abs_level, explicit = df['group'].iloc[0].split('_')
        concept = df['concept'].iloc[0]
        model_name = f.split('/')[2]

        fig, ax = plot_expertise(df)
        ax.set_title(f'Max AP per MLP Layer Type and Depth ({concept}/{abs_level}/{explicit}/{model_name})')

        out_fn = f'max_ap_per_mlp_layer_type_and_depth_{concept}_{abs_level}_{explicit}_{f.split("/")[2]}.png'
        out_dir = os.path.join(base_out_dir, model_name, abs_level, explicit)
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, out_fn))
        plt.close(fig)
        print(f"Saved plot to {out_fn}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate all plots for all the results in a given folder."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Directory containing the expertise CSV files.",
        default="wemb_exps/HuggingFaceTB/SmolLM3-3B-Base/basic_implicit"
    )

    args = parser.parse_args()
    main(args)