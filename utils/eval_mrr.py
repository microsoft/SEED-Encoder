import sys
sys.path += ["../"]
from utils.msmarco_eval import quality_checks_qids, compute_metrics, load_reference
import torch.distributed as dist
import gzip
import faiss
import numpy as np
from data.process_fn import dual_process_fn
from tqdm import tqdm
import torch
import os
from utils.util import concat_key, is_first_worker, all_gather, StreamingDataset
from torch.utils.data import DataLoader
import pytrec_eval
import csv

def convert_to_string_id(result_dict):
    string_id_dict = {}

    # format [string, dict[string, val]]
    for k, v in result_dict.items():
        _temp_v = {}
        for inner_k, inner_v in v.items():
            _temp_v[str(inner_k)] = inner_v

        string_id_dict[str(k)] = _temp_v

    return string_id_dict

def embedding_inference(args, path, model, fn, bz, num_workers=2, is_query=True):
    f = open(path, encoding="utf-8")
    model = model.module if hasattr(model, "module") else model
    sds = StreamingDataset(f, fn)
    loader = DataLoader(sds, batch_size=bz, num_workers=1)
    emb_list, id_list = [], []
    model.eval()
    # a=0
    # for i, batch in tqdm(enumerate(loader), desc="Eval", disable=args.local_rank not in [-1, 0]):
    #     a=i
    # print('????',a)
    for i, batch in tqdm(enumerate(loader), desc="Eval", disable=args.local_rank not in [-1, 0]):
        #print('???',i)
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0].long(
            ), "attention_mask": batch[1].long()}
            idx = batch[3].long()
            if is_query:
                embs = model.query_emb(**inputs)
            else:
                embs = model.body_emb(**inputs)
            if len(embs.shape) == 3:
                B, C, E = embs.shape
                # [b1c1, b1c2, b1c3, b1c4, b2c1 ....]
                embs = embs.view(B*C, -1)
                idx = idx.repeat_interleave(C)

            assert embs.shape[0] == idx.shape[0]
            emb_list.append(embs.detach().cpu().numpy())
            id_list.append(idx.detach().cpu().numpy())
    f.close()
    emb_arr = np.concatenate(emb_list, axis=0)
    id_arr = np.concatenate(id_list, axis=0)

    return emb_arr, id_arr


def parse_top_dev(input_path, qid_col, pid_col):
    ret = {}
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            cells = line.strip().split("\t")
            qid = int(cells[qid_col])
            pid = int(cells[pid_col])
            if qid not in ret:
                ret[qid] = []
            ret[qid].append(pid)
    return ret


def search_knn(xq, xb, k, distance_type=faiss.METRIC_L2):
    """ wrapper around the faiss knn functions without index """
    nq, d = xq.shape
    nb, d2 = xb.shape
    assert d == d2

    I = np.empty((nq, k), dtype='int64')
    D = np.empty((nq, k), dtype='float32')

    if distance_type == faiss.METRIC_L2:
        heaps = faiss.float_maxheap_array_t()
        heaps.k = k
        heaps.nh = nq
        heaps.val = faiss.swig_ptr(D)
        heaps.ids = faiss.swig_ptr(I)
        faiss.knn_L2sqr(
            faiss.swig_ptr(xq), faiss.swig_ptr(xb),
            d, nq, nb, heaps
        )
    elif distance_type == faiss.METRIC_INNER_PRODUCT:
        heaps = faiss.float_minheap_array_t()
        heaps.k = k
        heaps.nh = nq
        heaps.val = faiss.swig_ptr(D)
        heaps.ids = faiss.swig_ptr(I)
        faiss.knn_inner_product(
            faiss.swig_ptr(xq), faiss.swig_ptr(xb),
            d, nq, nb, heaps
        )
    return D, I


def get_topk_restricted(q_emb, psg_emb_arr, pid_dict, psg_ids, pid_subset, top_k):
    subset_ix = np.array([pid_dict[x]
                          for x in pid_subset if x != -1 and x in pid_dict])
    if len(subset_ix) == 0:
        _D = np.ones((top_k,))*-128
        _I = (np.ones((top_k,))*-1).astype(int)
        return _D, _I
    else:
        sub_emb = psg_emb_arr[subset_ix]
        _D, _I = search_knn(q_emb, sub_emb, top_k,
                            distance_type=faiss.METRIC_INNER_PRODUCT)
        return _D.squeeze(), psg_ids[subset_ix[_I]].squeeze()  # (top_k,)


def passage_dist_eval(args, model, tokenizer):
    base_path = args.data_dir
    passage_path = os.path.join(base_path, "collection.tsv")
    queries_path = os.path.join(base_path, "queries.dev.small.tsv")

    def fn(line, i):
        return dual_process_fn(line, i, tokenizer, args)

    top1000_path = os.path.join(base_path, "top1000.dev.tsv")
    top1k_qid_pid = parse_top_dev(top1000_path, qid_col=0, pid_col=1)

    mrr_ref_path = os.path.join(base_path, "qrels.dev.small.tsv")
    ref_dict = load_reference(mrr_ref_path)

    reranking_mrr, full_ranking_mrr = combined_dist_eval(
        args, model, queries_path, passage_path, fn, fn, top1k_qid_pid, ref_dict)
    return reranking_mrr, full_ranking_mrr


def passage_dist_eval_last(args, model, tokenizer):
    base_path = args.data_dir
    passage_path = os.path.join(base_path, "collection.tsv")
    queries_path = os.path.join(base_path, "queries.dev.small.tsv")

    def fn(line, i):
        return dual_process_fn(line, i, tokenizer, args)

    top1000_path = os.path.join(base_path, "top1000.dev.tsv")
    top1k_qid_pid = parse_top_dev(top1000_path, qid_col=0, pid_col=1)

    mrr_ref_path = os.path.join(base_path, "qrels.dev.small.tsv")
    ref_dict = load_reference(mrr_ref_path)



    # query_positive_id_path = os.path.join(base_path, "dev-qrel.tsv")
    # dev_query_positive_id = {}

    # with open(query_positive_id_path, 'r', encoding='utf8') as f:
    #     tsvreader = csv.reader(f, delimiter="\t")
    #     for [topicid, docid, rel] in tsvreader:
    #         topicid = int(topicid)
    #         docid = int(docid)
    #         if topicid not in dev_query_positive_id:
    #             dev_query_positive_id[topicid] = {}
    #         dev_query_positive_id[topicid][docid] = int(rel)
    dev_query_positive_id = {}
    for item in ref_dict:
        if item not in dev_query_positive_id:
            dev_query_positive_id[item]={}
        dev_query_positive_id[item][ref_dict[item]]=1

    recall_1000 = combined_dist_eval_last(
        args, model, queries_path, passage_path, fn, fn, top1k_qid_pid, ref_dict,dev_query_positive_id)



    return recall_1000



def combined_dist_eval_last(args, model, queries_path, passage_path, query_fn, psg_fn, topk_dev_qid_pid, ref_dict,dev_query_positive_id):
    # get query/psg embeddings here
    eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    query_embs, query_ids = embedding_inference(
        args, queries_path, model, query_fn, eval_batch_size, 1, True)
    query_pkl = {"emb": query_embs, "id": query_ids}
    all_query_list = all_gather(query_pkl)
    query_embs = concat_key(all_query_list, "emb")
    query_ids = concat_key(all_query_list, "id")
    print(query_embs.shape, query_ids.shape)
    psg_embs, psg_ids = embedding_inference(
        args, passage_path, model, psg_fn, eval_batch_size, 2, False)
    print(psg_embs.shape)

    top_k = 1000
    D, I = search_knn(query_embs, psg_embs, top_k,
                      distance_type=faiss.METRIC_INNER_PRODUCT)

    I = psg_ids[I]
    
    

    # qids_to_ranked_candidate_passages = {} 
    # for query_idx in range(len(I_nearest_neighbor)): 
    #     seen_pid = set()
    #     query_id = query_embedding2id[query_idx]
    #     prediction[query_id] = {}

    #     top_ann_pid = I_nearest_neighbor[query_idx].copy()
    #     selected_ann_idx = top_ann_pid[:topN]
    #     rank = 0
        
    #     if query_id in qids_to_ranked_candidate_passages:
    #         pass    
    #     else:
    #         # By default, all PIDs in the list of 1000 are 0. Only override those that are given
    #         tmp = [0] * 1000
    #         qids_to_ranked_candidate_passages[query_id] = tmp
                
    #     for idx in selected_ann_idx:
    #         pred_pid = passage_embedding2id[idx]
            
    #         if not pred_pid in seen_pid:
    #             # this check handles multiple vector per document
    #             qids_to_ranked_candidate_passages[query_id][rank]=pred_pid
    #             Atotal += 1
    #             if pred_pid not in dev_query_positive_id[query_id]:
    #                 Alabeled += 1
    #             # if rank < 10:
    #             #     total += 1
    #             #     if pred_pid not in dev_query_positive_id[query_id]:
    #             #         labeled += 1
    #             rank += 1
    #             prediction[query_id][pred_pid] = -rank
    #             seen_pid.add(pred_pid)

    # use out of the box evaluation script
    

    D2 = D[:, :1000]
    I2 = I[:, :1000]
    # full mrr
    recall_1000 = compute_mrr_last(D2, I2, query_ids, ref_dict,dev_query_positive_id)

    del psg_embs
    torch.cuda.empty_cache()
    dist.barrier()
    return recall_1000


def compute_mrr_last(D, I, qids, ref_dict,dev_query_positive_id):
    knn_pkl = {"D": D, "I": I}
    all_knn_list = all_gather(knn_pkl)
    mrr = 0.0
    final_recall=0.0
    if is_first_worker():
        prediction = {}
        D_merged = concat_key(all_knn_list, "D", axis=1)
        I_merged = concat_key(all_knn_list, "I", axis=1)
        print(D_merged.shape, I_merged.shape)
        # we pad with negative pids and distance -128 - if they make it to the top we have a problem
        idx = np.argsort(D_merged, axis=1)[:, ::-1][:, :1000]
        sorted_I = np.take_along_axis(I_merged, idx, axis=1)
        candidate_dict = {}
        for i, qid in enumerate(qids):
            seen_pids = set()
            if qid not in candidate_dict:
                prediction[qid] = {}
                #candidate_dict[qid] = [0]*1000
            j = 0
            for pid in sorted_I[i]:
                if pid >= 0 and pid not in seen_pids:
                    #candidate_dict[qid][j] = pid
                    prediction[qid][pid] =  -(j+1)#-rank
                    j += 1
                    seen_pids.add(pid)

        # allowed, message = quality_checks_qids(ref_dict, candidate_dict)
        # if message != '':
        #     print(message)

        # mrr_metrics = compute_metrics(ref_dict, candidate_dict)
        # mrr = mrr_metrics["MRR @10"]
        # print(mrr)
        evaluator = pytrec_eval.RelevanceEvaluator(
        convert_to_string_id(dev_query_positive_id), {'recall'})

        eval_query_cnt = 0
        recall = 0
        topN=1000
        result = evaluator.evaluate(convert_to_string_id(prediction))
        for k in result.keys():
            eval_query_cnt += 1
            recall += result[k]["recall_"+str(topN)]

        
        final_recall = recall / eval_query_cnt
        print('final_recall: ',final_recall)



    return final_recall





def combined_dist_eval(args, model, queries_path, passage_path, query_fn, psg_fn, topk_dev_qid_pid, ref_dict):
    # get query/psg embeddings here
    eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    query_embs, query_ids = embedding_inference(
        args, queries_path, model, query_fn, eval_batch_size, 1, True)
    query_pkl = {"emb": query_embs, "id": query_ids}
    all_query_list = all_gather(query_pkl)
    query_embs = concat_key(all_query_list, "emb")
    query_ids = concat_key(all_query_list, "id")
    print(query_embs.shape, query_ids.shape)
    psg_embs, psg_ids = embedding_inference(
        args, passage_path, model, psg_fn, eval_batch_size, 2, False)
    print(psg_embs.shape)

    top_k = 100
    D, I = search_knn(query_embs, psg_embs, top_k,
                      distance_type=faiss.METRIC_INNER_PRODUCT)
    I = psg_ids[I]

    # compute reranking and full ranking mrr here
    # topk_dev_qid_pid is used for computing reranking mrr
    pid_dict = dict([(p, i) for i, p in enumerate(psg_ids)])
    arr_data = []
    d_data = []
    for i, qid in enumerate(query_ids):
        q_emb = query_embs[i:i+1]
        pid_subset = topk_dev_qid_pid[qid]
        ds, top_pids = get_topk_restricted(
            q_emb, psg_embs, pid_dict, psg_ids, pid_subset, 10)
        arr_data.append(top_pids)
        d_data.append(ds)
    _D = np.array(d_data)
    _I = np.array(arr_data)

    # reranking mrr
    reranking_mrr = compute_mrr(_D, _I, query_ids, ref_dict)
    D2 = D[:, :100]
    I2 = I[:, :100]
    # full mrr
    full_ranking_mrr = compute_mrr(D2, I2, query_ids, ref_dict)
    del psg_embs
    torch.cuda.empty_cache()
    dist.barrier()
    return reranking_mrr, full_ranking_mrr


def compute_mrr(D, I, qids, ref_dict):
    knn_pkl = {"D": D, "I": I}
    all_knn_list = all_gather(knn_pkl)
    mrr = 0.0
    if is_first_worker():
        D_merged = concat_key(all_knn_list, "D", axis=1)
        I_merged = concat_key(all_knn_list, "I", axis=1)
        print(D_merged.shape, I_merged.shape)
        # we pad with negative pids and distance -128 - if they make it to the top we have a problem
        idx = np.argsort(D_merged, axis=1)[:, ::-1][:, :10]
        sorted_I = np.take_along_axis(I_merged, idx, axis=1)
        candidate_dict = {}
        for i, qid in enumerate(qids):
            seen_pids = set()
            if qid not in candidate_dict:
                candidate_dict[qid] = [0]*1000
            j = 0
            for pid in sorted_I[i]:
                if pid >= 0 and pid not in seen_pids:
                    candidate_dict[qid][j] = pid
                    j += 1
                    seen_pids.add(pid)

        allowed, message = quality_checks_qids(ref_dict, candidate_dict)
        if message != '':
            print(message)

        mrr_metrics = compute_metrics(ref_dict, candidate_dict)
        mrr = mrr_metrics["MRR @10"]
        print(mrr)
    return mrr
