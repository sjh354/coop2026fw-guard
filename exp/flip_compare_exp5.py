import pandas as pd

models = ['llamaguard', 'polyguard', 'sguard']
rows = []
for m in models:
    for lang, path in [('ko', f'results/ko_probe_{m}.csv'), ('en', f'results/en_probe_{m}.csv')]:
        df = pd.read_csv(path)
        baseline = df[df['variant_type'] == '원문'].set_index('base_id')['pred_label']
        df['baseline_pred'] = df['base_id'].map(baseline)
        fl = df.dropna(subset=['pred_label', 'baseline_pred'])
        fl = fl[fl['variant_type'] != '원문']
        fl = fl.assign(flip=fl['pred_label'] != fl['baseline_pred'])
        rate = fl.groupby('variant_type')['flip'].mean()
        for vt, r in rate.items():
            rows.append({'model': m, 'lang': lang, 'variant_type': vt, 'flip_rate': round(r, 4),
                         'n': int((fl['variant_type'] == vt).sum())})

out = pd.DataFrame(rows)
out.to_csv('results/final/flip_rate_en_vs_ko.csv', index=False)
print(out.pivot_table(index=['model', 'variant_type'], columns='lang', values='flip_rate'))
