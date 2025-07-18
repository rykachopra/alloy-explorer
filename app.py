import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr
from plotly.subplots import make_subplots
from functools import lru_cache
import networkx as nx
from flask import Flask
from waitress import serve

_ = go.Figure(layout=dict(template='plotly'))

DF_URL = "https://www.dropbox.com/scl/fi/ghz31iz0ujpgw25bz9qy5/Dataset_VisContest_Rapid_Alloy_development_v3.xlsx?rlkey=7ftblzur9dnmkw9h7z5q7vfsq&st=tn70gyv5&dl=1"
df = (
    pd.read_excel(DF_URL, engine="openpyxl")
      .dropna(subset=['CSC'])
      .reset_index(drop=True)
)
COLOR_COL = 'CSC'
MAX_POINTS = 500

all_feats = [c for c in df.columns if c != COLOR_COL]

drop_all_nan = [c for c in all_feats if df[c].isna().all()]
if drop_all_nan:
    df.drop(columns=drop_all_nan, inplace=True)
    all_feats = [c for c in df.columns if c != COLOR_COL]

imputer = SimpleImputer(strategy='mean')
df[all_feats] = imputer.fit_transform(df[all_feats])

full_pca = PCA(n_components=2)
coords_all = full_pca.fit_transform(df[all_feats])
df['PC1_all'], df['PC2_all'] = coords_all[:, 0], coords_all[:, 1]

orig_prop_feats = [
    'YS(MPa)', 'hardness(Vickers)', 'CTEvol(1/K)(20.0-300.0?C)',
    'Density(g/cm3)', 'Volume(m3/mol)', 'El.conductivity(S/m)',
    'El. resistivity(ohm m)', 'heat capacity(J/(mol K))',
    'Therm.conductivity(W/(mK))', 'Therm.diffusivity(m2/s)',
    'Therm.resistivity(mK/W)', 'Linear thermal expansion (1/K)(20.0-300.0?C)',
    'Technical thermal expansion (1/K)(20.0-300.0?C)'
]
mech_feats = [f for f in ['YS(MPa)', 'hardness(Vickers)', 'Density(g/cm3)'] if f in df.columns]
thermo_feats = [f for f in orig_prop_feats if f in df.columns and f not in mech_feats]

categories = {
    'scrap':  {'label': 'Scrap alloys',        'feats': ['KS1295[%]', '6082[%]', '2024[%]', 'bat-box[%]', '3003[%]', '4032[%]']},
    'comp':   {'label': 'Composition',          'feats': ['Al', 'Si', 'Cu', 'Ni', 'Mg', 'Mn', 'Fe', 'Cr', 'Ti', 'Zr', 'V', 'Zn']},
    'micro':  {'label': 'Microstructure',       'feats': [
        'Vf_FCC_A1','Vf_DIAMOND_A4','Vf_AL15SI2M4','Vf_AL3X','Vf_AL6MN','Vf_MG2ZN3',
        'Vf_AL3NI2','Vf_AL3NI_D011','Vf_AL7CU4NI','Vf_AL2CU_C16','Vf_Q_ALCUMGSI','Vf_AL7CU2FE',
        'Vf_MG2SI_C1','Vf_AL9FE2SI2','Vf_AL18FE2MG7SI10','T_FCC_A1','T_DIAMOND_A4',
        'T_AL15SI2M4','T_AL3X','T_AL6MN','T_MG2ZN3','T_AL3NI2','T_AL3NI_D011',
        'T_AL7CU4NI','T_AL2CU_C16','T_Q_ALCUMGSI','T_AL7CU2FE','T_MG2SI_C1','T_AL9FE2SI2',
        'T_AL18FE2MG7SI10','T(liqu)','T(sol)','eut. frac.[%]','eut. T (?C)','delta_T',
        'delta_T_FCC','delta_T_Al15Si2M4','delta_T_Si'
    ]},
    'mech':   {'label': 'Mechanical Props',     'feats': mech_feats},
    'thermo': {'label': 'Thermophysical Props', 'feats': thermo_feats}
}
for key, cat in categories.items():
    cat['feats'] = [f for f in cat['feats'] if f in df.columns]

for key, cat in categories.items():
    feats = cat['feats']
    if feats:
        pca = PCA(n_components=2)
        pts = pca.fit_transform(df[feats])
        df[f'PC1_{key}'], df[f'PC2_{key}'] = pts[:, 0], pts[:, 1]
        ld = pd.DataFrame(pca.components_.T, index=feats,
                          columns=['PC1 loading','PC2 loading'])
        cat['loadings_df'] = ld.reset_index().rename(columns={'index':'Feature'})

highlight_colors = {'scrap':'red','comp':'blue','micro':'green','mech':'purple','thermo':'orange'}

MAX_DEFAULT = min(MAX_POINTS, len(df))
binned_df = df[all_feats].round(1)

@lru_cache(maxsize=256)
def get_binned_unique_idx(vars_tuple: tuple):
    return tuple(binned_df[list(vars_tuple)].drop_duplicates().index)

@lru_cache(maxsize=256)
def get_corr(var1: str, var2: str):
    return pearsonr(df[var1], df[var2])
server = Flask(__name__)
app = dash.Dash(
    __name__,
    server=server,
    suppress_callback_exceptions=True
)

title = html.H2('PCA & Scatter Dashboard')
selector = html.Div([
    html.Label('Select alloy:'),
    dcc.Dropdown(
        id='alloy-selector',
        options=[{'label': f'Alloy {i}', 'value': i} for i in df.index]
    )
], style={'width':'200px','margin':'10px'})
slider = html.Div([
    html.Label('Category PCA Sample Size:'),
    dcc.Slider(
        id='category-sample-slider',
        min=50, max=len(df), step=50, value=MAX_DEFAULT,
        marks={i: str(i) for i in range(0, len(df)+1, max(1, len(df)//5))},
        tooltip={'always_visible': True}, updatemode='drag'
    )
], style={'padding':'10px','width':'80%'})


tabs = []

tabs.append(
    dcc.Tab(label='Category PCAs', children=html.Div([
        slider,
        html.Div(style={'display':'flex'}, children=[
            html.Div(style={'flex':'1','padding':'10px'}, children=[
                html.H3(cat['label']),
                dcc.Dropdown(
                    id=f'{key}-selector',
                    options=[{'label':f,'value':f} for f in cat['feats']],
                    multi=True, value=cat['feats'][:3]
                ),
                dcc.Dropdown(
                    id=f'{key}-rotation',
                    options=[{'label':f,'value':f} for f in cat['feats']],
                    placeholder='Rotate...'
                ),
                dcc.Graph(id=f'{key}-scatter'),
                dash_table.DataTable(
                    id=f'{key}-loadings',
                    columns=[{'name':c,'id':c} for c in cat['loadings_df'].columns],
                    data=cat['loadings_df'].to_dict('records')
                )
            ]) for key, cat in categories.items() if f'PC1_{key}' in df.columns
        ])
    ]))
)

tabs.append(dcc.Tab(label='All-Features PCA', children=dcc.Graph(id='linked-all')))

tabs.append(
    dcc.Tab(label='DeltaT_FCC vs Any', children=html.Div([
        html.Label('Feature:'),
        dcc.Dropdown(
            id='feature-selector',
            options=[{'label':f,'value':f} for f in all_feats if f!='delta_T_FCC'],
            value=[f for f in all_feats if f!='delta_T_FCC'][0]
        ),
        dcc.Graph(id='deltatfcc-scatter')
    ], style={'padding':'10px'}))
)

tabs.append(
    dcc.Tab(label='Correlation Matrix', children=html.Div([
        html.Div(style={'display':'flex','gap':'10px'}, children=[
            html.Div(style={'flex':1}, children=[
                html.Label('Scrap alloys'),
                dcc.Dropdown(
                    id='corr-vars-scrap',
                    options=[{'label':f,'value':f} for f in categories['scrap']['feats']],
                    multi=True, placeholder='Select scrap…'
                )
            ]),
            html.Div(style={'flex':1}, children=[
                html.Label('Composition'),
                dcc.Dropdown(
                    id='corr-vars-comp',
                    options=[{'label':f,'value':f} for f in categories['comp']['feats']],
                    multi=True, placeholder='Select composition…'
                )
            ]),
            html.Div(style={'flex':1}, children=[
                html.Label('Microstructure'),
                dcc.Dropdown(
                    id='corr-vars-micro',
                    options=[{'label':f,'value':f} for f in categories['micro']['feats']],
                    multi=True, placeholder='Select microstructure…'
                )
            ]),
            html.Div(style={'flex':1}, children=[
                html.Label('Mechanical Props'),
                dcc.Dropdown(
                    id='corr-vars-mech',
                    options=[{'label':f,'value':f} for f in categories['mech']['feats']],
                    multi=True, placeholder='Select mechanical…'
                )
            ]),
            html.Div(style={'flex':1}, children=[
                html.Label('Thermophysical Props'),
                dcc.Dropdown(
                    id='corr-vars-thermo',
                    options=[{'label':f,'value':f} for f in categories['thermo']['feats']],
                    multi=True, placeholder='Select thermophysical…'
                )
            ]),
        ]),
        dcc.Graph(id='corr-heatmap')
    ], style={'padding':'10px'}))
)

tabs.append(
    dcc.Tab(label='Correlation Network', children=html.Div([
        html.Label('Variables to link with CSC:'),
        dcc.Dropdown(
            id='net-vars',
            options=[{'label':f,'value':f} for f in all_feats if f!=COLOR_COL],
            multi=True, value=all_feats[:4]
        ),
        dcc.Graph(id='corr-network')
    ], style={'padding':'10px'}))
)

app.layout = html.Div([title, selector, dcc.Tabs(tabs)])
keys = [key for key in categories if f'PC1_{key}' in df.columns]

@app.callback(
    [Output(f'{key}-scatter','figure') for key in keys],
    [Input(f'{key}-selector','value') for key in keys] +
    [Input(f'{key}-rotation','value') for key in keys] +
    [Input(f'{key}-scatter','selectedData') for key in keys] +
    [Input(f'{key}-scatter','clickData') for key in keys] +
    [Input('category-sample-slider','value'),
     Input('alloy-selector','value')]
)
def update_category_plots(*args):
    n = len(keys)
    sel_feats  = args[:n]
    rot_feats  = args[n:2*n]
    sel_data   = args[2*n:3*n]
    clk_data   = args[3*n:4*n]
    sample_size, alloy_idx = args[-2], args[-1]

    all_pts = set()
    for sd, cd in zip(sel_data, clk_data):
        all_pts |= {pt['pointIndex'] for pt in (sd or {}).get('points', [])}
        all_pts |= {pt['pointIndex'] for pt in (cd or {}).get('points', [])}

    figs = []
    for i, key in enumerate(keys):
        feats = categories[key]['feats']
        dfp = df[[COLOR_COL] + feats].copy()
        dfp['PC1'], dfp['PC2'] = df[f'PC1_{key}'], df[f'PC2_{key}']

        if sample_size < len(dfp):
            mask_ext = (dfp[COLOR_COL] > 1) | (dfp[COLOR_COL] < 0.4)
            extremes = dfp[mask_ext]
            middle   = dfp[~mask_ext]
            if len(extremes) >= sample_size:
                dfp = extremes.sample(sample_size, random_state=0)
            else:
                needed = sample_size - len(extremes)
                dfp = pd.concat([extremes, middle.sample(needed, random_state=0)],
                                ignore_index=True)

        if rot_feats[i]:
            ld_df = categories[key]['loadings_df']
            r = ld_df[ld_df['Feature']==rot_feats[i]].iloc[0]
            theta = np.arctan2(r['PC1 loading'], r['PC2 loading'])
            frames = []
            for step, t in enumerate(np.linspace(0, theta, 20)):
                cos_t, sin_t = np.cos(t), np.sin(t)
                frames.append(go.Frame(
                    data=[go.Scatter(
                        x=dfp['PC1']*cos_t + dfp['PC2']*sin_t,
                        y=-dfp['PC1']*sin_t + dfp['PC2']*cos_t,
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=dfp[COLOR_COL],
                            colorscale=px.colors.diverging.Tealrose,
                            cmin=dfp[COLOR_COL].min(),
                            cmax=dfp[COLOR_COL].max()
                        )
                    )],
                    name=f'frame{step}'
                ))
            dfp['x'], dfp['y'] = dfp['PC1'], dfp['PC2']
            fig = go.Figure(
                data=[go.Scatter(
                    x=dfp['x'], y=dfp['y'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=dfp[COLOR_COL],
                        colorscale=px.colors.diverging.Tealrose,
                        cmin=dfp[COLOR_COL].min(),
                        cmax=dfp[COLOR_COL].max()
                    ),
                    hovertemplate='<b>%{pointIndex}</b><br>x: %{x}<br>y: %{y}<extra></extra>'
                )],
                frames=frames
            )
            fig.update_layout(
                updatemenus=[dict(
                    type='buttons', showactive=False,
                    buttons=[dict(
                        label='▶ Play',
                        method='animate',
                        args=[None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True, 'transition': {'duration': 0}
                        }]
                    )]
                )]
            )
        else:
            dfp['x'], dfp['y'] = dfp['PC1'], dfp['PC2']
            fig = px.scatter(
                dfp, x='x', y='y',
                color=COLOR_COL,
                color_continuous_scale=px.colors.diverging.Tealrose,
                hover_data=feats,
                template='plotly_white'
            )

        if i != len(keys) - 1:
            fig.update_coloraxes(showscale=False)

        scale = max(np.ptp(dfp['x']), np.ptp(dfp['y'])) * 0.7
        for feat in sel_feats[i] or []:
            v = categories[key]['loadings_df'].query("Feature==@feat").iloc[0]
            ox, oy = v['PC1 loading']*scale, v['PC2 loading']*scale
            fig.add_shape(type='line', x0=0, y0=0, x1=ox, y1=oy,
                          line=dict(color='black', width=2))
            fig.add_annotation(x=ox, y=oy, text=feat, showarrow=False)

        if all_pts:
            sel_d = dfp.iloc[list(all_pts)]
            fig.add_trace(go.Scatter(
                x=sel_d['x'], y=sel_d['y'], mode='markers',
                marker=dict(size=12, color='rgba(0,0,0,0)',
                            line=dict(width=2, color='black')),
                showlegend=False
            ))

        if alloy_idx is not None and i == len(keys) - 1:
            ax, ay = dfp.iloc[alloy_idx]['x'], dfp.iloc[alloy_idx]['y']
            fig.add_trace(go.Scatter(
                x=[ax], y=[ay], mode='markers',
                marker=dict(size=14, color='magenta',
                            line=dict(width=2, color='magenta')),
                name=f'Alloy {alloy_idx}'
            ))

        fig.update_layout(dragmode='lasso', height=450)
        figs.append(fig)

    return figs

@app.callback(
    Output('linked-all','figure'),
    [Input(f'{key}-scatter','selectedData') for key in keys] +
    [Input('alloy-selector','value')]
)
def update_global(*args):
    sels, alloy_idx = args[:-1], args[-1]
    fig = px.scatter(
        df, x='PC1_all', y='PC2_all',
        color=COLOR_COL,
        color_continuous_scale=px.colors.diverging.Tealrose,
        title='Global PCA',
        hover_data=all_feats,
        template='plotly_white'
    )
    for key, sel in zip(keys, sels):
        if sel and 'points' in sel:
            pts = [p['pointIndex'] for p in sel['points']]
            sel_d = df.iloc[pts]
            fig.add_trace(go.Scatter(
                x=sel_d['PC1_all'], y=sel_d['PC2_all'], mode='markers',
                marker=dict(color=highlight_colors[key], size=8),
                name=categories[key]['label']
            ))
    if alloy_idx is not None:
        fig.add_trace(go.Scatter(
            x=[df.loc[alloy_idx,'PC1_all']],
            y=[df.loc[alloy_idx,'PC2_all']],
            mode='markers',
                    marker=dict(size=14, color='magenta',
                                line=dict(width=2, color='magenta')),
                    name=f'Alloy {alloy_idx}'
                ))
    fig.update_layout(dragmode='select')
    return fig

@app.callback(
    Output('deltatfcc-scatter','figure'),
    Input('feature-selector','value')
)
def update_scatter(feature):
    fig = px.scatter(
        df, x='delta_T_FCC', y=feature,
        color=COLOR_COL,
        color_continuous_scale=px.colors.diverging.Tealrose,
        hover_data=[feature],
        title=f"delta_T_FCC vs {feature}",
        template='plotly_white'
    )
    fig.update_layout(xaxis_title='delta_T_FCC', yaxis_title=feature)
    return fig

@app.callback(
    Output('corr-heatmap','figure'),
    Input('corr-vars-scrap','value'),
    Input('corr-vars-comp','value'),
    Input('corr-vars-micro','value'),
    Input('corr-vars-mech','value'),
    Input('corr-vars-thermo','value'),
)
def update_corr_heatmap(scrap, comp, micro, mech, thermo):
    selected = []
    for lst in (scrap, comp, micro, mech, thermo):
        if lst:
            selected.extend(lst)
    if len(selected) < 2:
        fig = go.Figure()
        fig.update_layout(
            title='Select ≥2 variables',
            xaxis={'visible':False}, yaxis={'visible':False}
        )
        return fig

    idx_all = get_binned_unique_idx(tuple(selected))
    if len(idx_all) > MAX_POINTS:
        idx_sample = list(np.random.choice(idx_all, MAX_POINTS, replace=False))
    else:
        idx_sample = list(idx_all)
    df_small = df.loc[idx_sample]

    n = len(selected)
    spacing = min(0.02, 1/(n-1)) if n>1 else 0
    fig = make_subplots(
        rows=n, cols=n,
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=spacing, vertical_spacing=spacing
    )

    for i in range(n):
        for j in range(n):
            if i > j:
                x = df_small[selected[j]]
                y = df_small[selected[i]]
                r, p = get_corr(selected[j], selected[i])
                fig.add_trace(
                    go.Scattergl(
                        x=x, y=y, mode='markers',
                        marker=dict(
                            size=6, opacity=0.6,
                            color=[abs(r)]*len(x),
                            colorscale='Blues', cmin=0, cmax=1,
                            showscale=False
                        )
                    ), row=i+1, col=j+1
                )
                fig.add_annotation(
                    x=0.5, y=0.9,
                    xref=f"x{i*n+j+1} domain",
                    yref=f"y{i*n+j+1} domain",
                    text=f"r={r:.2f}, p={p:.3f}",
                    showarrow=False,
                    font=dict(size=10, color='black')    
                )

    for i in range(n):
        for j in range(n):
            ax_id = i*n + j + 1
            xa = f'xaxis{ax_id if ax_id>1 else ""}'
            ya = f'yaxis{ax_id if ax_id>1 else ""}'
            if i < n-1:
                fig.layout[ya].showticklabels = False
            if j > 0:
                fig.layout[xa].showticklabels = False

    for k, var in enumerate(selected):
        bot = (n-1)*n + k + 1
        fig.layout[f'xaxis{bot if bot>1 else ""}'].title = var
        lef = k*n + 1
        fig.layout[f'yaxis{lef if lef>1 else ""}'].title = var

    fig.update_layout(
        width=700, height=700,
        title='Matrix Plot of ' + ', '.join(selected),
        template='plotly_white',
        margin=dict(l=80, r=80, t=80, b=80),
        dragmode='select',
    )
    fig.update_xaxes(tickfont=dict(size=11), title_font=dict(size=13))
    fig.update_yaxes(tickfont=dict(size=11), title_font=dict(size=13))

    return fig

@app.callback(
    Output('corr-network','figure'),
    Input('net-vars','value')
)
def update_corr_network(vars_selected):
    if not vars_selected:
        fig = go.Figure()
        fig.update_layout(
            title='Select ≥1 variable',
            xaxis={'visible': False}, yaxis={'visible': False}
        )
        return fig

    corr_series = df[vars_selected].apply(lambda col: df[COLOR_COL].corr(col)).dropna()

    G = nx.Graph()
    G.add_node('CSC')
    for v, r in corr_series.items():
        G.add_node(v)
        G.add_edge('CSC', v, weight=r)

    pos = nx.spring_layout(G, k=0.5, seed=0, iterations=50, weight=None)
    cmap = px.colors.diverging.RdBu
    def map_color(r):
        t   = (r + 1) / 2
        idx = int(t * (len(cmap)-1))
        return cmap[idx]

    fig = go.Figure()
    for u, v, d in G.edges(data=True):
        r = d['weight']
        x0,y0 = pos[u]
        x1,y1 = pos[v]
        fig.add_trace(go.Scatter(
            x=[x0,x1], y=[y0,y1],
            mode='lines',
            line=dict(color=map_color(r), width=max(1, abs(r)*8)),
            hoverinfo='text',
            hovertext=f"{v}: r = {r:.2f}",
            showlegend=False
        ))

    nodes     = ['CSC'] + corr_series.index.tolist()
    node_x    = [pos[n][0] for n in nodes]
    node_y    = [pos[n][1] for n in nodes]
    node_color= [1.0] + [corr_series[v] for v in corr_series.index]
    node_size = [20]  + [10 + abs(corr_series[v])*20 for v in corr_series.index]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=nodes, textposition='top center',
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale='RdBu',
            cmin=-1, cmax=1,
            showscale=True,
            colorbar=dict(title='r with CSC')
        ),
        hovertemplate="<b>%{text}</b><br>r = %{marker.color:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title='CSC-centric Correlation Network (all selected vars)',
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='white',
        height=600,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

if __name__ == '__main__':
    serve(server, host='127.0.0.1', port=8000)
