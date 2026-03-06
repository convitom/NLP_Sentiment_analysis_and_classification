from datasets import load_dataset

dataset = load_dataset("go_emotions")



dataset['train'].to_pandas().to_csv("D:\\USTH\\nlp\\final_prj\\data\\train.csv", index=False, encoding='utf-8')
dataset['test'].to_pandas().to_csv("D:\\USTH\\nlp\\final_prj\\data\\test.csv", index=False, encoding='utf-8')
dataset['validation'].to_pandas().to_csv("D:\\USTH\\nlp\\final_prj\\data\\val.csv", index=False, encoding='utf-8')
