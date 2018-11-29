import pickle as pkl
import pandas as pd

def handleFile(substr=''):
    with open('output/paramRes%s.pkl' % substr, 'rb') as f:
        res = pkl.load(f)
    with open('output/paramRes%sList.pkl' % substr, 'rb') as f:
        resList = pkl.load(f)
    res.insert(1, 'Class', 'RMSE')
    res = res.set_index([res.index.set_names(['$\lambda$']), 'Class'])
    dfTime = pd.DataFrame(index=res.index.set_levels(['Steps'],level='Class'), columns=res.columns)
    for key, value in resList.items():
        dfTime.loc[key] = len(value[0])
    res = res.append(dfTime).sort_index()
    with open('report/meta/paramRes%s.tex' % substr, 'w', encoding='utf8') as resFile:
        print(res.to_latex(multirow=True, escape=False, column_format='ll'+'r'*len(res.columns)), file=resFile)

handleFile()
handleFile('L3')
handleFile('LM')
