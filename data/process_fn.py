import torch
import numpy as np

def pad_ids(input_ids, attention_mask, token_type_ids, max_length, pad_token, mask_padding_with_zero, pad_token_segment_id, pad_on_left=False):
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1]
                          * padding_length) + attention_mask
        token_type_ids = ([pad_token_segment_id] *
                          padding_length) + token_type_ids
    else:
        input_ids += [pad_token] * padding_length
        attention_mask += [0 if mask_padding_with_zero else 1] * padding_length
        token_type_ids += [pad_token_segment_id] * padding_length

    return input_ids, attention_mask, token_type_ids


def dual_process_fn(line, i, tokenizer, args):
    features = []
    cells = line.split("\t")
    if len(cells) == 2:
        # this is for training and validation
        # id, passage = line
        mask_padding_with_zero = True
        pad_token_segment_id = 0
        pad_on_left = False

        text = cells[1].strip()
        if 'fairseq' not in args.train_model_type:
            input_id_a = tokenizer.encode(
                text, add_special_tokens=True, max_length=args.max_seq_length,)
            pad_token_id=tokenizer.pad_token_id
        elif 'fast' in args.train_model_type:
            text=text.lower()
            input_id_a=tokenizer.encode(text, add_special_tokens=True).ids[:args.max_seq_length]
            pad_token_id=1
        else:
            text=text.lower()
            input_id_a=list(np.array(tokenizer.encode(text)[:args.max_seq_length]))
            pad_token_id=1
        token_type_ids_a = [0] * len(input_id_a)
        attention_mask_a = [
            1 if mask_padding_with_zero else 0] * len(input_id_a)
        input_id_a, attention_mask_a, token_type_ids_a = pad_ids(
            input_id_a, attention_mask_a, token_type_ids_a, args.max_seq_length, pad_token_id, mask_padding_with_zero, pad_token_segment_id, pad_on_left)
        features += [torch.tensor(input_id_a, dtype=torch.int), torch.tensor(
            attention_mask_a, dtype=torch.bool), torch.tensor(token_type_ids_a, dtype=torch.uint8)]
        qid = int(cells[0])
        features.append(qid)
    else:
        raise Exception(
            "Line doesn't have correct length: {0}. Expected 2.".format(str(len(cells))))
    return [features]

def dual_process_fn_doc(line, i, tokenizer, args):
    features = []
    cells = line.split("\t")
    if len(cells) == 4:
        # this is for training and validation
        # id, passage = line
        mask_padding_with_zero = True
        pad_token_segment_id = 0
        pad_on_left = False

        url = cells[1].rstrip()
        title = cells[2].rstrip()
        p_text = cells[3].rstrip()

        if 'fast' in args.train_model_type:
            text = url.lower() + " [SEP] " + title.lower() + " [SEP] " + p_text.lower()
        elif 'fairseq' in args.train_model_type:
            text = url + " </s> " + title + " </s> " + p_text
        else:
            text = url + " "+tokenizer.sep_token_id+" " + title + " "+tokenizer.sep_token_id+" " + p_text

        #text = cells[1].strip()
        text=text[:10000]
        if 'fairseq' not in args.train_model_type:
            input_id_a = tokenizer.encode(
                text, add_special_tokens=True, max_length=args.max_seq_length,)
            pad_token_id=tokenizer.pad_token_id
        elif 'fast' in args.train_model_type:
            #text=text.lower()
            input_id_a=tokenizer.encode(text, add_special_tokens=True).ids[:args.max_seq_length]
            pad_token_id=1
        else:
            text=text.lower()
            input_id_a=list(np.array(tokenizer.encode(text)[:args.max_seq_length]))
            pad_token_id=1
        token_type_ids_a = [0] * len(input_id_a)
        attention_mask_a = [
            1 if mask_padding_with_zero else 0] * len(input_id_a)
        input_id_a, attention_mask_a, token_type_ids_a = pad_ids(
            input_id_a, attention_mask_a, token_type_ids_a, args.max_seq_length, pad_token_id, mask_padding_with_zero, pad_token_segment_id, pad_on_left)
        features += [torch.tensor(input_id_a, dtype=torch.int), torch.tensor(
            attention_mask_a, dtype=torch.bool), torch.tensor(token_type_ids_a, dtype=torch.uint8)]
        qid = int(cells[0].strip('D'))
        features.append(qid)
        #print('doc: ',text)
    elif len(cells) == 2:
        mask_padding_with_zero = True
        pad_token_segment_id = 0
        pad_on_left = False

        text = cells[1].strip()
        if 'fairseq' not in args.train_model_type:
            input_id_a = tokenizer.encode(
                text, add_special_tokens=True, max_length=args.max_seq_length,)
            pad_token_id=tokenizer.pad_token_id
        elif 'fast' in args.train_model_type:
            text=text.lower()
            input_id_a=tokenizer.encode(text, add_special_tokens=True).ids[:args.max_seq_length]
            pad_token_id=1
        else:
            text=text.lower()
            input_id_a=list(np.array(tokenizer.encode(text)[:args.max_seq_length]))
            pad_token_id=1
        token_type_ids_a = [0] * len(input_id_a)
        attention_mask_a = [
            1 if mask_padding_with_zero else 0] * len(input_id_a)
        input_id_a, attention_mask_a, token_type_ids_a = pad_ids(
            input_id_a, attention_mask_a, token_type_ids_a, args.max_seq_length, pad_token_id, mask_padding_with_zero, pad_token_segment_id, pad_on_left)
        features += [torch.tensor(input_id_a, dtype=torch.int), torch.tensor(
            attention_mask_a, dtype=torch.bool), torch.tensor(token_type_ids_a, dtype=torch.uint8)]
        qid = int(cells[0])
        features.append(qid)
        #print('query: ',input_id_a)
    else:
        raise Exception(
            "Line doesn't have correct length: {0}. Expected 2.".format(str(len(cells))))
    return [features]


def triple_process_fn(line, i, tokenizer, args,data_type=1):
    features = []
    cells = line.split("\t")
    if len(cells) == 3:
        # this is for training and validation
        # query, positive_passage, negative_passage = line
        mask_padding_with_zero = True
        pad_token_segment_id = 0
        pad_on_left = False
        cell_index=0
        for text in cells:
            cell_index+=1
            if 'fairseq' not in args.train_model_type:
                input_id_a = tokenizer.encode(
                    text.strip(), add_special_tokens=True, max_length=args.max_seq_length,)
                pad_token_id=tokenizer.pad_token_id
                #print('???',input_id_a)
            elif 'fast' in args.train_model_type:
                #if data_type==1:
                if cell_index ==1 or data_type==1:
                    text=text.lower()

                # if getattr(args, "data_type", 1)==1:
                #     text=text.lower()
                # print('???',args.data_type)
                input_id_a=tokenizer.encode(text.strip(), add_special_tokens=True).ids[:args.max_seq_length]
                # if cell_index ==1:
                #     print('query: ',input_id_a)
                # else:
                #     print('cell_index: ',cell_index,input_id_a)
                pad_token_id=1
            else:
                text=text.lower()
                input_id_a=list(np.array( tokenizer.encode(text.strip())[:args.max_seq_length]))
                pad_token_id=1
            token_type_ids_a = [0] * len(input_id_a)
            attention_mask_a = [
                1 if mask_padding_with_zero else 0] * len(input_id_a)
            input_id_a, attention_mask_a, token_type_ids_a = pad_ids(
                input_id_a, attention_mask_a, token_type_ids_a, args.max_seq_length, pad_token_id, mask_padding_with_zero, pad_token_segment_id, pad_on_left)
            features += [torch.tensor(input_id_a, dtype=torch.int),
                         torch.tensor(attention_mask_a, dtype=torch.bool)]
    elif len(cells)==2:
        mask_padding_with_zero = True
        pad_token_segment_id = 0
        pad_on_left = False
        cell_index=0
        for text in cells:
            cell_index+=1
            if 'fairseq' not in args.train_model_type:
                input_id_a = tokenizer.encode(
                    text.strip(), add_special_tokens=True, max_length=args.max_seq_length,)
                pad_token_id=tokenizer.pad_token_id
            elif 'fast' in args.train_model_type:
                #if data_type==1:
                # if cell_index ==1 or data_type==1:
                #     text=text.lower()

                # if getattr(args, "data_type", 1)==1:
                #     text=text.lower()
                # print('???',args.data_type)
                input_id_a=tokenizer.encode(text.strip(), add_special_tokens=True).ids[:args.max_seq_length]
                #print('???',input_id_a)
                # if cell_index ==1:
                #     print('query: ',input_id_a)
                # else:
                #     print('cell_index: ',cell_index,input_id_a)
                pad_token_id=1
            else:
                text=text.lower()
                input_id_a=list(np.array( tokenizer.encode(text.strip())[:args.max_seq_length]))
                pad_token_id=1
            token_type_ids_a = [0] * len(input_id_a)
            attention_mask_a = [
                1 if mask_padding_with_zero else 0] * len(input_id_a)
            input_id_a, attention_mask_a, token_type_ids_a = pad_ids(
                input_id_a, attention_mask_a, token_type_ids_a, args.max_seq_length, pad_token_id, mask_padding_with_zero, pad_token_segment_id, pad_on_left)
            features += [torch.tensor(input_id_a, dtype=torch.int),
                         torch.tensor(attention_mask_a, dtype=torch.bool)]
    else:
        raise Exception(
            "Line doesn't have correct length: {0}. Expected 3.".format(str(len(cells))))
    return [features]


def triple2dual_process_fn(line, i, tokenizer, args):
    ret = []
    cells = line.split("\t")
    if len(cells) == 3:
        # this is for training and validation
        # query, positive_passage, negative_passage = line
        # return 2 entries per line, 1 pos + 1 neg
        mask_padding_with_zero = True
        pad_token_segment_id = 0
        pad_on_left = False
        pos_feats = []
        neg_feats = []

        for i, text in enumerate(cells):
            input_id_a = tokenizer.encode(
                text.strip(), add_special_tokens=True, max_length=args.max_seq_length,)
            token_type_ids_a = [0] * len(input_id_a)
            attention_mask_a = [
                1 if mask_padding_with_zero else 0] * len(input_id_a)
            input_id_a, attention_mask_a, token_type_ids_a = pad_ids(
                input_id_a, attention_mask_a, token_type_ids_a, args.max_seq_length, tokenizer.pad_token_id, mask_padding_with_zero, pad_token_segment_id, pad_on_left)
            if i == 0:
                pos_feats += [torch.tensor(input_id_a, dtype=torch.int),
                              torch.tensor(attention_mask_a, dtype=torch.bool)]
                neg_feats += [torch.tensor(input_id_a, dtype=torch.int),
                              torch.tensor(attention_mask_a, dtype=torch.bool)]
            elif i == 1:
                pos_feats += [torch.tensor(input_id_a, dtype=torch.int),
                              torch.tensor(attention_mask_a, dtype=torch.bool), 1]
            else:
                neg_feats += [torch.tensor(input_id_a, dtype=torch.int),
                              torch.tensor(attention_mask_a, dtype=torch.bool), 0]
        ret = [pos_feats, neg_feats]
    else:
        raise Exception(
            "Line doesn't have correct length: {0}. Expected 3.".format(str(len(cells))))
    return ret

