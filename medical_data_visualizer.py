import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1: Import data
df = pd.read_csv('medical_examination.csv')

# 2: Add 'overweight' column (BMI > 25)
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3: Normalize cholesterol and gluc (1 -> 0 [good], 2/3 -> 1 [bad])
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4: Draw Categorical Plot
def draw_cat_plot():
    # 5: Create DataFrame for cat plot using `pd.melt`
    df_cat = pd.melt(df,
                     id_vars=['cardio'],
                     value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    # 6: Group and reformat the data to show counts of each feature
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 8: Draw the catplot
    fig = sns.catplot(x='variable',
                      y='total',
                      hue='value',
                      col='cardio',
                      kind='bar',
                      data=df_cat).fig

    # 9: Save and return figure
    fig.savefig('catplot.png')
    return fig

# 10: Draw Heat Map
def draw_heat_map():
    # 11: Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12: Calculate the correlation matrix
    corr = df_heat.corr()

    # 13: Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14: Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15: Draw the heatmap
    sns.heatmap(corr,
                mask=mask,
                annot=True,
                fmt=".1f",
                center=0,
                vmax=0.3,
                vmin=-0.1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .5},
                ax=ax)

    # 16: Save and return figure
    fig.savefig('heatmap.png')
    return fig
