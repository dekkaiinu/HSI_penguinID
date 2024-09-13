import numpy as np

def calc_penguin_id(predict_scores):
    scores = np.array(predict_scores)
    ranks = np.argsort(-scores, axis=1)

    # init
    decision_labels = np.zeros_like(ranks[:, 0])
    pred_scores = np.zeros_like(scores[:, 0])

    sort_scores = np.sort(scores, axis=None)[::-1]
    for i in range(len(sort_scores)):
        if sort_scores[i] == 0:
            break
        attn_score_index = np.argwhere(scores==sort_scores[i])[0]
        attn_score_index = attn_score_index.reshape((2,))
        if not (attn_score_index[1] + 1) in decision_labels:
                if decision_labels[attn_score_index[0]] == 0:
                    decision_labels[attn_score_index[0]] = attn_score_index[1] + 1
                    pred_scores[attn_score_index[0]]  = sort_scores[i]
    decision_labels = decision_labels - 1
    
    pred_ids = decision_labels.tolist()
    return pred_ids, pred_scores.tolist()

def argmax_id(predict_scores):
    pred_ids = np.argmax(predict_scores, axis=1)
    pred_scores = np.max(predict_scores, axis=1)
    return pred_ids, pred_scores
