from IPython.display import display_html
import pandas as pd

def display_side_by_side(*args):
    html_str = ''
    for df in args:
        html_str += df.to_html()
    display_html(html_str.replace('table', 'table style="display:inline"'), raw=True)

def display_2d_results(results):
    dfs = []
    for score_key in results:
        df = pd.DataFrame(results[score_key]).set_index("Summarizer")
        styled_df = df.style.set_caption(score_key).format("{:.2f}")
        styled_df = styled_df.background_gradient(cmap="Blues", axis=None, low=0.0, high=1.0)
        
        dfs.append(styled_df)
    display_side_by_side(*dfs)
