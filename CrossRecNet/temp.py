from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

dic = {'sub12': {'pred': [1, 1, 0, 0, 0, 1, 0, 2, 2, 2, 2], 'truth': [1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2]}, 'sub13': {'pred': [1, 1], 'truth': [1, 1]}, '011': {'pred': [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 'truth': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, '022': {'pred': [1, 1, 0, 0, 0], 'truth': [1, 1, 0, 0, 0]}, '026': {'pred': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'truth': [0, 0, 0, 0, 0, 0, 0, 0, 0]}, 'sub03': {'pred': [0, 0, 0, 0, 2], 'truth': [0, 0, 0, 0, 2]}, 'sub05': {'pred': [1, 2, 2, 2, 2, 2], 'truth': [1, 2, 2, 2, 2, 2]}, '012': {'pred': [0, 0, 2], 'truth': [0, 0, 2]}, 's05': {'pred': [0, 0], 'truth': [0, 2]}, 's09': {'pred': [1, 2, 2, 2], 'truth': [1, 2, 2, 2]}, 's18': {'pred': [0, 0, 2, 2, 2, 2, 2], 'truth': [0, 0, 2, 2, 2, 2, 2]}, 's11': {'pred': [1, 1, 1, 0, 0, 0, 2], 'truth': [1, 1, 1, 0, 0, 0, 2]}, '010': {'pred': [0, 0, 0, 0], 'truth': [0, 0, 0, 0]}, '036': {'pred': [0], 'truth': [0]}, 'sub26': {'pred': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'truth': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, 's03': {'pred': [1, 1, 1, 1, 1, 1, 2, 1, 0, 1, 1, 0, 0, 2, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 2, 0, 2, 0, 2], 'truth': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2]}, 'sub09': {'pred': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 'truth': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]}, 's08': {'pred': [2, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'truth': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, 's20': {'pred': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 2], 'truth': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2]}, '021': {'pred': [0, 0], 'truth': [0, 0]}, 's01': {'pred': [1, 1, 0, 0, 0, 1], 'truth': [1, 1, 0, 0, 0, 2]}, '007': {'pred': [1, 1, 1, 1, 1, 1, 1, 1], 'truth': [1, 1, 1, 1, 1, 0, 2, 2]}, '032': {'pred': [0, 0, 0, 0], 'truth': [0, 0, 0, 0]}, 'sub15': {'pred': [1, 0, 2], 'truth': [1, 0, 2]}, 'sub06': {'pred': [1, 0, 2, 2], 'truth': [1, 0, 2, 2]}, '015': {'pred': [0, 0, 2], 'truth': [0, 0, 2]}, 's19': {'pred': [1, 2], 'truth': [1, 2]}, '009': {'pred': [0, 0, 0, 1], 'truth': [0, 0, 0, 2]}, 'sub25': {'pred': [0, 0, 0, 2, 2], 'truth': [0, 0, 0, 2, 2]}, 's06': {'pred': [0, 0, 2, 0], 'truth': [0, 0, 2, 2]}, '028': {'pred': [0, 2, 2], 'truth': [0, 2, 2]}, 'sub08': {'pred': [0], 'truth': [0]}, '033': {'pred': [2, 0, 0, 0, 0], 'truth': [1, 0, 0, 0, 0]}, '006': {'pred': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2], 'truth': [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2]}}
sub_list_truth = []
sub_list_pred = []
s_list_pred = []
s_list_truth = []
num_list_pred = []
num_list_truth = []
for i in dic.items():

    if i[0].startswith('sub'):
        sub_list_truth += i[1]['truth']
        sub_list_pred += i[1]['pred']

    elif i[0].startswith('s'):
        s_list_pred += i[1]['pred']
        s_list_truth += i[1]['truth']
    else:
        num_list_pred += i[1]['pred']
        num_list_truth += i[1]['truth']
#calculate UF1 and UAR
def UF1_UAR(truth, pred):
    UF1 = f1_score(truth, pred, average='macro')
    UAR = recall_score(truth, pred, average='macro')
    return UF1, UAR

UF1_sub, UAR_sub = UF1_UAR(sub_list_truth, sub_list_pred)
UF1_s, UAR_s = UF1_UAR(s_list_truth, s_list_pred)
UF1_num, UAR_num = UF1_UAR(num_list_truth, num_list_pred)
UF1_all, UAR_all = UF1_UAR(sub_list_truth + s_list_truth + num_list_truth, sub_list_pred + s_list_pred + num_list_pred)
print(f'UF1_sub: {UF1_sub}, UAR_sub: {UAR_sub}')
print(f'UF1_s: {UF1_s}, UAR_s: {UAR_s}')
print(f'UF1_num: {UF1_num}, UAR_num: {UAR_num}')
print(f'UF1_all: {UF1_all}, UAR_all: {UAR_all}')
