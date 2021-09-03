import sys
sys.path += ['../']
import torch
from torch import nn
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    BertModel,
    BertTokenizer,
    BertConfig,
    BertForSequenceClassification, 
)
import torch.nn.functional as F
from data.process_fn import triple_process_fn, triple2dual_process_fn


try:
    from fairseq.modules import (
        TransformerSentenceEncoder,
    )
except ImportError:
    raise ImportError('Please install fairseq to the fairseq version model')



from transformers import AutoTokenizer, AutoModel
from model.SEED_Encoder import SEEDEncoderConfig, SEEDTokenizer, SEEDEncoderForSequenceClassification,SEEDEncoderForMaskedLM




import torch.distributed as dist
def is_first_worker():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


class EmbeddingMixin:
    """
    Mixin for common functions in most embedding models. Each model should define its own bert-like backbone and forward.
    We inherit from RobertaModel to use from_pretrained 
    """
    def __init__(self, model_argobj):
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean
        print("Using mean:", self.use_mean)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        if isinstance(emb_all, tuple):
            if self.use_mean:
                return self.masked_mean(emb_all[0], mask)
            else:
                return emb_all[0][:, 0]
        else:
            #print('!!!',emb_all.shape)
            if self.use_mean:
                return self.masked_mean(emb_all, mask)
            else:
                #print('??? should be the first')
                return emb_all[:, 0]

    def query_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

    def body_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")


class NLL(EmbeddingMixin):
    def forward(
            self,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            is_query=True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        #print('???',q_embs.shape,a_embs.shape)

        logit_matrix = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1),
                                  (q_embs * b_embs).sum(-1).unsqueeze(1)], dim=1)  # [B, 2]

        lsm = F.log_softmax(logit_matrix, dim=1)

        loss = -1.0 * lsm[:, 0]
        return (loss.mean(),)


        


class NLL_MultiChunk(EmbeddingMixin):
    def forward(
            self,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            is_query=True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        [batchS, full_length] = input_ids_a.size()
        chunk_factor = full_length // self.base_len

        # special handle of attention mask -----
        attention_mask_body = attention_mask_a.reshape(
            batchS, chunk_factor, -1)[:, :, 0]  # [batchS, chunk_factor]
        inverted_bias = ((1 - attention_mask_body) * (-9999)).float()

        a12 = torch.matmul(
            q_embs.unsqueeze(1), a_embs.transpose(
                1, 2))  # [batch, 1, chunk_factor]
        logits_a = (a12[:, 0, :] + inverted_bias).max(dim=-
                                                      1, keepdim=False).values  # [batch]
        # -------------------------------------

        # special handle of attention mask -----
        attention_mask_body = attention_mask_b.reshape(
            batchS, chunk_factor, -1)[:, :, 0]  # [batchS, chunk_factor]
        inverted_bias = ((1 - attention_mask_body) * (-9999)).float()

        a12 = torch.matmul(
            q_embs.unsqueeze(1), b_embs.transpose(
                1, 2))  # [batch, 1, chunk_factor]
        logits_b = (a12[:, 0, :] + inverted_bias).max(dim=-
                                                      1, keepdim=False).values  # [batch]
        # -------------------------------------

        logit_matrix = torch.cat(
            [logits_a.unsqueeze(1), logits_b.unsqueeze(1)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        return (loss.mean(),)


class RobertaDot_NLL_LN(NLL, RobertaForSequenceClassification):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """

    def __init__(self, config, model_argobj=None):
        NLL.__init__(self, model_argobj)
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)





class RobertaDot_NLL_LN_fairseq(NLL,nn.Module):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """
    def __init__(self, config, model_argobj=None):
        nn.Module.__init__(self)
        NLL.__init__(self, model_argobj)
        
        self.encoder=TransformerSentenceEncoder(
                padding_idx=1,
                vocab_size=50265,
                num_encoder_layers=12,
                embedding_dim=768,
                ffn_embedding_dim=3072,
                num_attention_heads=12,
                dropout=0.1,
                attention_dropout=0.1,
                activation_dropout=0.0,
                layerdrop=0.0,
                max_seq_len=512,
                num_segments=0,
                encoder_normalize_before=True,
                apply_bert_init=True,
                activation_fn="gelu",
                q_noise=0.0,
                qn_block_size=8,
        )
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)

    def query_emb(self, input_ids, attention_mask):
        #print('???input_ids',input_ids.shape)
        outputs1, _ = self.encoder(input_ids)#[-1].transpose(0,1)
        #print('???',outputs1)
        outputs1=outputs1[-1].transpose(0,1)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)

    def from_pretrained(self, model_path):
        model_dict = self.state_dict()
        save_model=torch.load(model_path, map_location=lambda storage, loc: storage)
        #print(save_model['model'].keys())
        pretrained_dict= {}
        if 'model' in save_model.keys():
            #save_model['model']
            for name in save_model['model']:
                if  'lm_head' not in name and 'decode' not in name:
                    pretrained_dict['encoder'+name[24:]]=save_model['model'][name]
            assert len(model_dict)-4==len(pretrained_dict)
        else:
            for name in save_model:
                pretrained_dict[name[7:]]=save_model[name]
            assert len(model_dict)==len(pretrained_dict)

        #print(model_dict.keys())
        print('load model.... ',len(model_dict),len(pretrained_dict))
        print(pretrained_dict.keys())
        
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


class RobertaDot_NLL_LN_fairseq_fast(NLL,nn.Module):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """
    def __init__(self, config, model_argobj=None):
        nn.Module.__init__(self)
        NLL.__init__(self, model_argobj)
        
        self.encoder=TransformerSentenceEncoder(
                padding_idx=1,
                vocab_size=32769,
                num_encoder_layers=12,
                embedding_dim=768,
                ffn_embedding_dim=3072,
                num_attention_heads=12,
                dropout=0.1,
                attention_dropout=0.1,
                activation_dropout=0.0,
                layerdrop=0.0,
                max_seq_len=512,
                num_segments=0,
                encoder_normalize_before=True,
                apply_bert_init=True,
                activation_fn="gelu",
                q_noise=0.0,
                qn_block_size=8,
        )
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)

    def query_emb(self, input_ids, attention_mask):
        outputs1, _ = self.encoder(input_ids)#[-1].transpose(0,1)
        outputs1=outputs1[-1].transpose(0,1)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))

        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)

    #tikenid pad
    def from_pretrained(self, model_path):
        model_dict = self.state_dict()
        save_model=torch.load(model_path, map_location=lambda storage, loc: storage)
        pretrained_dict= {}
        if 'model' in save_model.keys():
            #save_model['model']
            for name in save_model['model']:
                if 'lm_head' not in name and 'encoder' in name and 'decode' not in name:
                    pretrained_dict['encoder'+name[24:]]=save_model['model'][name]
            assert len(model_dict)-4==len(pretrained_dict), (len(model_dict),len(pretrained_dict),model_dict.keys(),pretrained_dict.keys())
        else:
            print('load finetuned checkpoints...')
            for name in save_model:
                pretrained_dict[name[7:]]=save_model[name]
            assert len(model_dict)==len(pretrained_dict)

        #print(model_dict.keys())
        print('load model.... ',len(model_dict),len(pretrained_dict))
        print(pretrained_dict.keys())
        
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        #pass









class RobertaDot_CLF_ANN_NLL_MultiChunk(NLL_MultiChunk, RobertaDot_NLL_LN):
    def __init__(self, config):
        RobertaDot_NLL_LN.__init__(self, config)
        self.base_len = 512

    def body_emb(self, input_ids, attention_mask):
        [batchS, full_length] = input_ids.size()
        chunk_factor = full_length // self.base_len

        input_seq = input_ids.reshape(
            batchS,
            chunk_factor,
            full_length //
            chunk_factor).reshape(
            batchS *
            chunk_factor,
            full_length //
            chunk_factor)
        attention_mask_seq = attention_mask.reshape(
            batchS,
            chunk_factor,
            full_length //
            chunk_factor).reshape(
            batchS *
            chunk_factor,
            full_length //
            chunk_factor)

        outputs_k = self.roberta(input_ids=input_seq,
                                 attention_mask=attention_mask_seq)
        compressed_output_k = self.embeddingHead(
            outputs_k[0])  # [batch, len, dim]
        compressed_output_k = self.norm(compressed_output_k[:, 0, :])

        [batch_expand, embeddingS] = compressed_output_k.size()
        complex_emb_k = compressed_output_k.reshape(
            batchS, chunk_factor, embeddingS)

        return complex_emb_k  # size [batchS, chunk_factor, embeddingS]

class RobertaDot_CLF_ANN_NLL_MultiChunk_fairseq_fast(NLL_MultiChunk, RobertaDot_NLL_LN_fairseq_fast):
    def __init__(self, config):
        RobertaDot_NLL_LN_fairseq_fast.__init__(self, config)
        self.base_len = 512

    def body_emb(self, input_ids, attention_mask):
        [batchS, full_length] = input_ids.size()
        chunk_factor = full_length // self.base_len

        input_seq = input_ids.reshape(
            batchS,
            chunk_factor,
            full_length //
            chunk_factor).reshape(
            batchS *
            chunk_factor,
            full_length //
            chunk_factor)
        outputs_k, _= self.encoder(input_seq)
        outputs_k=outputs_k[-1].transpose(0,1)       
        compressed_output_k = self.embeddingHead(
            outputs_k)  # [batch, len, dim]
        compressed_output_k = self.norm(compressed_output_k[:, 0, :])

        [batch_expand, embeddingS] = compressed_output_k.size()
        complex_emb_k = compressed_output_k.reshape(
            batchS, chunk_factor, embeddingS)

        return complex_emb_k  # size [batchS, chunk_factor, embeddingS]


class SEEDEncoderDot_NLL_LN(NLL, SEEDEncoderForMaskedLM):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """
    def __init__(self, config, model_argobj=None):
        NLL.__init__(self, model_argobj)
        SEEDEncoderForMaskedLM.__init__(self, config)
        self.embeddingHead = nn.Linear(config.encoder_embed_dim, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)

    def query_emb(self, input_ids, attention_mask=None):
        outputs1 = self.seed_encoder.encoder(input_ids)


        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask=None):
        return self.query_emb(input_ids, attention_mask)



class HFBertEncoder(BertModel):
    def __init__(self, config):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.init_weights()
    @classmethod
    def init_encoder(cls, args, dropout: float = 0.1):
        cfg = BertConfig.from_pretrained("bert-base-uncased")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained("bert-base-uncased", config=cfg)
    def forward(self, input_ids, attention_mask):
        hidden_states = None
        sequence_output, pooled_output = super().forward(input_ids=input_ids,
                                                         attention_mask=attention_mask)
        pooled_output = sequence_output[:, 0, :]
        return sequence_output, pooled_output, hidden_states
    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class BiEncoder(nn.Module):
    """ Bi-Encoder model component. Encapsulates query/question and context/passage encoders.
    """
    def __init__(self, args):
        super(BiEncoder, self).__init__()
        self.question_model = HFBertEncoder.init_encoder(args)
        self.ctx_model = HFBertEncoder.init_encoder(args)
    def query_emb(self, input_ids, attention_mask):
        sequence_output, pooled_output, hidden_states = self.question_model(input_ids, attention_mask)
        return pooled_output
    def body_emb(self, input_ids, attention_mask):
        sequence_output, pooled_output, hidden_states = self.ctx_model(input_ids, attention_mask)
        return pooled_output
    def forward(self, query_ids, attention_mask_q, input_ids_a = None, attention_mask_a = None, input_ids_b = None, attention_mask_b = None):
        if input_ids_b is None:
            q_embs = self.query_emb(query_ids, attention_mask_q)
            a_embs = self.body_emb(input_ids_a, attention_mask_a)
            return (q_embs, a_embs)
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)
        logit_matrix = torch.cat([(q_embs*a_embs).sum(-1).unsqueeze(1), (q_embs*b_embs).sum(-1).unsqueeze(1)], dim=1) #[B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0*lsm[:,0]
        return (loss.mean(),)
        

# --------------------------------------------------
ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys()) 
        for conf in (
            RobertaConfig,
        ) if hasattr(conf,'pretrained_config_archive_map')
    ),
    (),
)


default_process_fn = triple_process_fn


class MSMarcoConfig:
    def __init__(self, name, model, process_fn=default_process_fn, use_mean=True, tokenizer_class=RobertaTokenizer, config_class=RobertaConfig):
        self.name = name
        self.process_fn = process_fn
        self.model_class = model
        self.use_mean = use_mean
        self.tokenizer_class = tokenizer_class
        self.config_class = config_class


configs = [
    MSMarcoConfig(name="rdot_nll",
                model=RobertaDot_NLL_LN,
                use_mean=False,
                ),
    MSMarcoConfig(name="rdot_nll_multi_chunk",
                model=RobertaDot_CLF_ANN_NLL_MultiChunk,
                use_mean=False,
                ),
    MSMarcoConfig(name="dpr",
                model=BiEncoder,
                tokenizer_class=BertTokenizer,
                config_class=BertConfig,
                use_mean=False,
                ),


    MSMarcoConfig(name="rdot_nll_fairseq",
                model=RobertaDot_NLL_LN_fairseq,
                use_mean=False,
                #config_class=,
                ),
    MSMarcoConfig(name="rdot_nll_fairseq_fast",
                model=RobertaDot_NLL_LN_fairseq_fast,
                use_mean=False,
                #config_class=,
                ),

    MSMarcoConfig(name="seeddot_nll",
                model=SEEDEncoderDot_NLL_LN,
                use_mean=False,
                tokenizer_class=SEEDTokenizer,
                config_class=SEEDEncoderConfig,
                ),

    MSMarcoConfig(name="rdot_nll_multi_chunk_fairseq_fast",
                model=RobertaDot_CLF_ANN_NLL_MultiChunk_fairseq_fast,
                use_mean=False,
                ),
]

MSMarcoConfigDict = {cfg.name: cfg for cfg in configs}
