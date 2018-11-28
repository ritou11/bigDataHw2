import pickle as pkl
import pandas as pd

def handleFile(substr=''):
    with open('output/paramRes%s.pkl' % substr, 'rb') as f:
        res = pkl.load(f)
    with open('output/paramRes%sList.pkl' % substr, 'rb') as f:
        resList = pkl.load(f)
    with open('report/meta/paramRes%s.tex' % substr, 'w', encoding='utf8') as resFile:
        print(res.to_latex(), file=resFile)

handleFile()
handleFile('L3')
handleFile('LM')
