import pandas as pd
from pathlib import Path
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from keras.preprocessing.sequence import pad_sequences



def read_review(datatype="TRAIN", path=Path("..")):

    if datatype == "TRAIN":
        file = path / datatype / "Train_reviews.csv"
    elif datatype == "TEST":
        file = path / datatype / "Test_reviews.csv"
    df = pd.read_csv(file, sep=",")
    return df


def read_label(datatype="TRAIN", path=Path("..")):

    file = path / datatype / "Train_labels.csv"
    df = pd.read_csv(file, sep=",")
    return df


def test01():
    """
    count    3229.000000
    mean       21.700836
    std        10.894433
    min         1.000000
    25%        13.000000
    50%        20.000000
    75%        28.000000
    max        69.000000
    """
    df = read_review()
    print(df["Reviews"].str.len().describe())


def test02():
    df = read_label()
    idx1 = df["AspectTerms"] != "_"
    idx2 = df["OpinionTerms"] != "_"

    print(df.shape, df[idx1 & idx2].shape)


def test03():
    MAX_LEN = 48
    df_review = read_review()
    df_label = read_label()
    token_label = np.zeros((df_review.shape[0], MAX_LEN))

    for rowid, row in df_label.iterrows():

        idx = row["id"] - 1
        if row["A_start"] != " " and row["A_end"] != " ":
            token_label[idx, int(row["A_start"]) : int(row["A_end"])] = 1

        if row["O_start"] != " " and row["O_end"] != " ":
            token_label[idx, int(row["O_start"]) : int(row["O_end"])] = 2

    return 0


def test04():
    df_review = read_review()
    df_label = read_label()
    df_review_test = read_review("TEST")
    df_pred = pd.read_csv("../output/pred_0827.csv", sep=",", header=None)

    train_terms = df_label.apply(
        lambda x: (
            x["AspectTerms"] + x["OpinionTerms"],
            (x["Categories"], x["Polarities"]),
        ),
        axis=1,
    )

    train_terms = dict(list(train_terms))
    pred_terms = df_pred.apply(lambda x: (x[1] + x[2], (x[3], x[4])), axis=1)
    pred_terms = dict(list(pred_terms))

    for pred_term, pred_attr in pred_terms.items():
        if pred_term in train_terms:
            print(pred_term, pred_attr)
            # if pred_attr != train_terms[pred_term]:
            # print(pred_term, pred_attr, train_terms[pred_term])

    return


def test05():

    df_review = read_review()
    df_label = read_label()

    enhance_text(df_review, df_label)


def enhance_text(df_review, df_label):
    idx = df_review["Reviews"].map(lambda x: len(x.split("，"))) > 2

    df_review_enhance = df_review[idx]
    df_label_enhance = df_label_enhance[idx]


def test06():

    df = read_review()
    print(df["Reviews"].str.len().describe())
    df = read_review("TEST")
    print(df["Reviews"].str.len().describe())


    
def text_to_seq(texts,tokenizer,maxlen=48):
    
    input_texts = []
    for idx, text in enumerate(texts):
        text_id = [
            tokenizer.vocab.get(token, tokenizer.vocab["[UNK]"]) for token in text
        ]
        input_texts.append(text_id)

        assert len(text_id) == len(text)

    input_ids = pad_sequences(
        input_texts, maxlen=maxlen, dtype="long", truncating="post", padding="post"
    )

    attention_masks = np.array([[float(i > 0) for i in ii] for ii in input_ids])

    return input_ids, attention_masks
    
def encode_seq(maxlen=48):
    # id AspectTerms A_start A_end OpinionTerms O_start O_end Categories Polarities
    df_review = read_review()
    df_label = read_label()
    label_cp = list(
        df_label.apply(
            lambda x: "-".join([x["Categories"], x["Polarities"]]), axis=1
        ).drop_duplicates()
    )

    label_to_id = dict([(x, y + 1) for x, y in zip(label_cp, range(len(label_cp)))])

    term_to_id = {
        "A-B": 1,
        "A-I": 2,
        "A-E": 3,
        "A-S": 4,
        "O-B": 5,
        "O-I": 6,
        "O-E": 7,
        "O-S": 8,
    }

    def encode_term(pos, label):

        seq = np.zeros((maxlen,), dtype=np.int32)
        for (s, e) in pos:
            if e - s == 1:
                seq[s] = term_to_id["%s-S" % label]
            else:
                seq[s] = term_to_id["%s-B" % label]
                seq[e - 1] = term_to_id["%s-E" % label]
                for p in range(s + 1, e - 1):
                    seq[p] = term_to_id["%s-I" % label]
        return seq.reshape((1, -1))

    def encode_label(pos_a, pos_o, label):

        seq = np.zeros((maxlen,), dtype=np.int32)
        for (s, e), l in zip(pos_a, label):
            if s == " " or int(e) >= maxlen:
                continue
            s = int(s)
            e = int(e)
            if e - s == 1:
                seq[s] = label_to_id[l]
            else:
                seq[s] = label_to_id[l]
                seq[e - 1] = label_to_id[l]
                for p in range(s + 1, e - 1):
                    seq[p] = label_to_id[l]

        for (s, e), l in zip(pos_o, label):
            if s == " " or int(e) >= maxlen:
                continue
            s = int(s)
            e = int(e)
            if e - s == 1:
                seq[s] = label_to_id[l]
            else:
                seq[s] = label_to_id[l]
                seq[e - 1] = label_to_id[l]
                for p in range(s + 1, e - 1):
                    seq[p] = label_to_id[l]
        return seq.reshape((1, -1))

    seq_A = df_label.groupby("id").apply(
        lambda x: encode_term(
            [
                (int(s), int(e))
                for s, e in zip(x["A_start"], x["A_end"])
                if s != " " and int(e) < maxlen
            ],
            "A",
        )
    )

    seq_O = df_label.groupby("id").apply(
        lambda x: encode_term(
            [
                (int(s), int(e))
                for s, e in zip(x["O_start"], x["O_end"])
                if s != " " and int(e) < maxlen
            ],
            "O",
        )
    )

    seq_CP = df_label.groupby("id").apply(
        lambda x: encode_label(
            [(s, e) for s, e in zip(x["A_start"], x["A_end"])],
            [(s, e) for s, e in zip(x["O_start"], x["O_end"])],
            [(e + "-" + p) for e, p in zip(x["Categories"], x["Polarities"])],
        )
    )
    
    seq_id = np.array(df_label.groupby("id").apply(lambda x: list(x['id'])[0]).to_list())
    seq_A = np.vstack(seq_A)
    seq_O = np.vstack(seq_O)
    seq_AO = seq_A + seq_O  # 前提是A和O没有重合

    seq_CP = np.vstack(seq_CP)

    id_to_label = dict([(v, k) for k, v in label_to_id.items()])
    id_to_term = dict([(v, k) for k, v in term_to_id.items()])
    return seq_id, seq_AO, seq_CP, id_to_label, id_to_term


def pair_a_to_o(a_terms, o_terms):
    viewpoints = []
    temp_idx_a = []
    for idx_a, a_t in enumerate(a_terms):
        center_a = (a_t[2] + a_t[3]) / 2
        dist = []
        if len(o_terms) == 0:
            break
        for o_t in o_terms:
            center_o = (o_t[2] + o_t[3]) / 2
            d = abs(center_a - center_o) + (a_t[1] != o_t[1]) * 5
            dist.append(d)
        idx_o = np.array(dist).argmin()

        if a_t[1] == o_terms[idx_o][1]:  # 只将类别相同的A和O合并
            viewpoints.append((a_t[0], o_terms[idx_o][0], a_t[1]))
            del o_terms[idx_o]
            # del a_terms[idx_a]
            temp_idx_a.append(idx_a)

    for o_t in o_terms:
        viewpoints.append(("_", o_t[0], o_t[1]))

    for idx_a, a_t in enumerate(a_terms):
        if idx_a in temp_idx_a:
            continue
        viewpoints.append((a_t[0], "_", a_t[1]))

    return viewpoints


def decode_seq(seq_id, seq_AO, seq_CP, id_to_label, id_to_term, text_review):
    max_len = seq_AO.shape[1]
    seq_idx = np.arange(max_len)
    assert seq_AO.shape[0] == seq_CP.shape[0] == len(text_review)
    viewpoints = []
    for id, s_ao, s_cp, text in zip(seq_id, seq_AO, seq_CP, text_review):
        idx_ab = seq_idx[np.where(s_ao == 1, True, False)]
        idx_ae = seq_idx[np.where(s_ao == 3, True, False)]
        idx_ai = seq_idx[np.where(s_ao == 4, True, False)]
        idx_ob = seq_idx[np.where(s_ao == 5, True, False)]
        idx_oe = seq_idx[np.where(s_ao == 7, True, False)]
        idx_oi = seq_idx[np.where(s_ao == 8, True, False)]

        a_terms, o_terms = [], []
        for i_b, i_e in zip(idx_ab, idx_ae):
            if i_b >= i_e + 1:
                continue
            label = max(s_cp[i_b : i_e + 1])
            a_terms.append((text[i_b : i_e + 1], label, i_b, i_e + 1))

        for i_i in idx_ai:
            label = max(s_cp[i_i : i_i + 1])
            a_terms.append((text[i_i : i_i + 1], label, i_i, i_i + 1))

        for i_b, i_e in zip(idx_ob, idx_oe):
            if i_b >= i_e + 1:
                continue
            label = max(s_cp[i_b : i_e + 1])
            o_terms.append((text[i_b : i_e + 1], label, i_b, i_e + 1))

        for i_i in idx_oi:
            label = max(s_cp[i_i : i_i + 1])
            o_terms.append((text[i_i : i_i + 1], label, i_i, i_i + 1))

        vp = pair_a_to_o(a_terms.copy(), o_terms.copy())
        vp = [(str(id),) + v[:2] + tuple(id_to_label[v[2]].split("-")) for v in vp if v[2] > 0]
        viewpoints.extend(vp)

        if len(a_terms) > 0 and len(a_terms) < len(o_terms):
            pass

        if len(a_terms) > 0 and len(a_terms) == len(o_terms):
            pass

        if len(idx_ai) > 0 or len(idx_oi) > 0:
            pass

        if len(a_terms) > 0 and len(a_terms) > len(o_terms):
            pass

    return viewpoints


def cal_metrics(pred_vp, true_vp):

    
    p = len(pred_vp)
    g = len(true_vp)
    s = len(set(pred_vp) & set(true_vp))
        
    precision = s / (p+0.00001)
    recall = s / (g+0.000001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    print(
        precision,
        recall,
        f1,
        "\n",
    )
    
    return f1


if __name__ == "__main__":
    
    BERT_PT_PATH = "/data1/1906_nl2sql/baseline-cai/data/chinese_L-12_H-768_A-12/"
    df_review = read_review()
    df_label = read_label()
    
    seq_id, seq_AO, seq_CP, id_to_label, id_to_term = encode_seq(maxlen=48)
    tokenizer = BertTokenizer.from_pretrained(BERT_PT_PATH, do_lower_case=True)
    input_ids, attention_masks = text_to_seq(list(df_review["Reviews"]),tokenizer,maxlen=48)

    true_vp = [
        (
            row["id"],
            row["AspectTerms"],
            row["OpinionTerms"],
            row["Categories"],
            row["Polarities"],
        )
        for rowid, row in df_label.iterrows()
    ]
    pred_vp = decode_seq(
        seq_id, seq_AO, seq_CP, id_to_label, id_to_term, list(df_review["Reviews"])
    )
    
    cal_metrics(pred_vp, true_vp)
    
    