# -*- encoding:utf-8 -*-
from tkinter import NO
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from modules.RNNEncoders import RNNEncoder
from modules.info_nce import InfoNCE
from modules.supcontrast import SupConLoss
from modules.transformers import Transformer, LayerNorm
from transformers import T5EncoderModel,ASTModel,\
OpenAIGPTModel,GPT2Model,BertModel, ASTConfig, AutoModel,EncodecModel,EncodecConfig,Wav2Vec2Config, Wav2Vec2Model



from config import DEVICE,hidden_dict

# print('load package successfuly')

class Model(nn.Module):
    def __init__(self,
                 args, embeddings=None):
        super().__init__()

        ###prompt
        self.args = args
        self.dataset_name = self.args.dataset

        self.hidden_size = args.hidden_size
        self.text_projection = nn.Parameter(torch.empty(self.args.transformer_width, self.args.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.option = args.option

        self.acoustic_encoder_name = self.args.acoustic_encoder_name
        if self.acoustic_encoder_name == 'rnn':
            self.audio_encoder = RNNEncoder(
                    in_size=self.args.d_ain,
                    hidden_size=self.args.d_ah,
                    out_size=self.args.d_aout,
                    num_layers=self.args.n_layer,
                    dropout=self.args.dropout_t if self.args.n_layer > 1 else 0.3,
                    bidirectional=self.args.bidirectional
                )
        elif self.acoustic_encoder_name == 'ast':
            configuration = ASTConfig()
            self.audio_encoder = ASTModel(configuration)
            # self.audio_encoder = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        elif self.acoustic_encoder_name == 'wav2vec':
            # Initializing a Wav2Vec2 facebook/wav2vec2-base-960h style configuration
            # configuration = Wav2Vec2Config()
            # self.audio_encoder = Wav2Vec2Model(configuration)
            self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        elif args.acoustic_encoder_name == 'encodec':
            # configuration = EncodecConfig(sampling_rate=args.sampling_rate)
            # self.audio_encoder = EncodecModel(configuration)
            self.audio_encoder = EncodecModel.from_pretrained("facebook/encodec_24khz")
        else:
            print('invalid acoustic encoder name:{}'.format(self.acoustic_encoder_name))

            
        self.text_encoder_name = args.text_encoder_name
        if self.text_encoder_name == 'transformer':
            self.vocab_size = self.args.vocab_size
            # self.emb = nn.Embedding.from_pretrained(torch.tensor(embeddings, dtype=torch.float))
            self.token_embedding = nn.Embedding(self.vocab_size, self.args.transformer_width)
            self.positional_embedding = nn.Parameter(torch.empty(self.args.context_length, self.args.transformer_width))
            self.ln_final = LayerNorm(self.args.transformer_width)

            self.text_encoder = Transformer(
                width=self.args.transformer_width,
                layers=self.args.transformer_layers,
                heads=self.args.transformer_heads,
                attn_mask=self.build_attention_mask()
            )
            self.initialize_parameters()
        elif self.text_encoder_name == 'bert':
            self.text_encoder = BertModel.from_pretrained("google-bert/bert-base-uncased")
        elif self.text_encoder_name == 't5':
            self.text_encoder = T5EncoderModel.from_pretrained("google-t5/t5-base")
        elif args.text_encoder_name== 'llama':
            access_token = ''
            self.text_encoder = AutoModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",token=access_token)
        elif args.text_encoder_name == 'mistral':
            access_token = ''
            self.text_encoder = AutoModel.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1',token=access_token)
        else:
            print('text encoder name:{}'.format(self.text_encoder_name))


        def set_params_for_layers(name, fine_layers):
            if len(fine_layers) == 0:
                return False

            for task_param in fine_layers:
                if task_param in name:
                    return True
        ### freeze acoustic encoder
        # breakpoint()
        if args.freeze_audio:
            self.audio_fine_layers = args.audio_fine_layers
            # print('---------------audio encoder------------')
            if self.acoustic_encoder_name == 'ast':
                tmp = []
                for i, layer in enumerate(self.audio_fine_layers):
                    print('layer.{}'.format(i+10))
                    tmp.append('layer.{}'.format(i+10))
                self.audio_fine_layers = tmp
            for name, param in self.audio_encoder.named_parameters():
                # print(name)
                if set_params_for_layers(name, self.audio_fine_layers):
                    # print('set:{}'.format(name))
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        else:
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = False

        if not args.freeze_text:
            self.text_fine_layers = args.text_fine_layers
            for name, param in self.text_encoder.named_parameters():
                if set_params_for_layers(name, self.text_fine_layers):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for name, param in self.text_encoder.named_parameters():
                param.requires_grad = False


        self.infonce = InfoNCE()
        self.supcontrast = SupConLoss()

        self.ce_loss = nn.CrossEntropyLoss()
        # self.haptic_discriminator = nn.Sequential(
        #     nn.Linear(hidden_size*2, hidden_size),
        #     nn.Dropout(p=0.5),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, 3),
        # )
        if args.text_encoder_name in ['llama', 'mistral']:
            self.linear_trans = nn.Linear(4096, self.hidden_size)

        if self.option=='match' and self.dataset_name=='gendata':

            self.match_discriminator = nn.Sequential(
                nn.Linear(hidden_size*2, hidden_size),
                nn.Dropout(p=0.5),
                nn.ReLU(),
                nn.Linear(hidden_size, 2))
            ###match -> vote=1 and unmatch -> vote = 0

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.args.context_length, self.args.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.args.transformer_width ** -0.5) * ((2 * self.args.transformer_layers) ** -0.5)
        attn_std = self.args.transformer_width ** -0.5
        fc_std = (2 * self.args.transformer_width) ** -0.5
        for block in self.text_encoder.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.args.transformer_width ** -0.5)

    # @property
    # def dtype(self):
    #     return self.

    def forward(self, texts, audio, alens=None, category=None, label=None, mode='train'):

        ###encode raw senttence
        ###generated haptic signal, category refers to vote=1 or vote=0

        ### sentence_embedding (bs, t_dim) => (bs, 768)
        ### sen_feature (bs, seq_len, d_model)
        ### audio_feature ()
        # print('category:{}'.format(category))
        ### text encoder
        if self.text_encoder_name == 'transformer':
            tag_feature = self.token_embedding(texts).to(DEVICE)  # [batch_size, n_ctx, d_model]

            tag_feature = tag_feature + self.positional_embedding
            tag_feature = tag_feature.permute(1, 0, 2)  # NLD -> LND
            tag_feature = self.text_encoder(tag_feature)
            tag_feature = tag_feature.permute(1, 0, 2)  # LND -> NLD
            text_feature = self.ln_final(tag_feature)

            text_features = text_feature[torch.arange(text_feature.shape[0]), texts.argmax(dim=-1)] @ self.text_projection
        elif self.text_encoder_name == 't5':
            input_ids = texts
            text_features = self.text_encoder(input_ids=input_ids).last_hidden_state
            text_features = torch.mean(text_features,dim=1)
        elif self.text_encoder_name == 'bert':
            input_ids = texts
            text_features = self.text_encoder(**input_ids).last_hidden_state
            text_features = torch.mean(text_features,dim=1)
        elif self.text_encoder_name == 'gpt':
            input_ids = texts
            text_features = self.text_encoder(input_ids=input_ids).last_hidden_state
            text_features = torch.mean(text_features,dim=1)
        elif self.text_encoder_name in ['llama','mistral']:
            input_ids = texts
            text_features = self.text_encoder(**input_ids, output_hidden_states=False, return_dict=True).last_hidden_state
            text_features = torch.mean(text_features,dim=1)

            text_features = self.linear_trans(text_features)
        
        ###acoustic encoder
        if self.acoustic_encoder_name == 'rnn':
            acoustic_features, _ = self.audio_encoder(audio,alens.cpu())
        elif self.acoustic_encoder_name == 'ast':
            acoustic_features = self.audio_encoder(audio['input_values']).pooler_output
        elif self.acoustic_encoder_name == 'wav2vec':
            # print('audio input values:{}'.format(audio['input_values']))
            acoustic_features = self.audio_encoder(audio['input_values'].float()).last_hidden_state
            acoustic_features = torch.mean(acoustic_features, dim=1)
        elif self.acoustic_encoder_name == 'encodec':
            acoustic_features = self.audio_encoder(audio["input_values"], audio["padding_mask"]).audio_values
            acoustic_features = acoustic_features.squeeze(dim=1)[:,:self.hidden_size]
            # acoustic_features = torch.mean(acoustic_features, dim=1)   
        else:
            print('acoustic encoder name:{}'.format(self.acoustic_encoder_name))

        # print('acoustic_shape:{}'.format(acoustic_features.shape))
        if len(acoustic_features.shape)==1:
            acoustic_features = acoustic_features.unsqueeze(dim=0)

        acoustic_features = acoustic_features / acoustic_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # print('text_feature.shape:{},acoustic_feature.shape:{}'.format(text_features.shape, acoustic_features.shape))

        def contrastive_loss(acoustic_features, text_features):
            #(global_batch_size, dim) * (global_batch_size, dim).t()->(global_batch_size, global_batch_size)

            # cosine similarity as logits
            global_size = acoustic_features.shape[0]
            logits_per_acoustic = self.logit_scale.exp() * acoustic_features @ text_features.t()
            logits_per_text = logits_per_acoustic.t()

            # n_signal = global_size // 3
            # diff = global_size - 3*n_signal
            # target_label = []

            # for i in range(n_signal):
            #     target_label.append(3*[i])
            
            # if diff != 0:
            #     target_label.append(diff*[n_signal])
            # target_label = [item for sublist in target_label for item in sublist]
            ### constrastive learning for each haptic signal with 3 categories

            target_label = list(range(global_size))

            labels = torch.LongTensor(target_label).flatten().to(DEVICE)
            # print('label.shape:{}'.format(labels.shape))
            loss = (F.cross_entropy(logits_per_acoustic, labels) +
                    F.cross_entropy(logits_per_text, labels)) / 2


            # shape = [global_batch_size, global_batch_size]
            # infonce_loss = self.infonce(text_features, acoustic_features)
            return loss, logits_per_acoustic, logits_per_text
        
        def sup_contrastive_loss(text_features, acoustic_features=None, 
                                 labels=None, mask=None, contrast_mode='all', temperature=0.07, 
                                 base_temperature=0.1):
            ### label <- signal id + category (including augumentation)
            ### text_feature 

            sample_features = torch.cat([text_features, acoustic_features], dim=-1)

            if len(sample_features.shape) < 3:
                sample_features = sample_features.unsqueeze(dim=1)
            if len(sample_features.shape) > 3:
                sample_features = sample_features.view(sample_features.shape[0], sample_features.shape[1], -1)

            batch_size = sample_features.shape[0]
            if labels is not None and mask is not None:
                raise ValueError('Cannot define both `labels` and `mask`')
            elif labels is None and mask is None:
                mask = torch.eye(batch_size, dtype=torch.float32).to(DEVICE)
            elif labels is not None:
                labels = labels.contiguous().view(-1, 1)
                if labels.shape[0] != batch_size:
                    raise ValueError('Num of labels does not match num of features')
                mask = torch.eq(labels, labels.T).float().to(DEVICE)
            else:
                mask = mask.float().to(DEVICE)

            contrast_count = sample_features.shape[1]
            contrast_feature = torch.cat(torch.unbind(sample_features, dim=1), dim=0)
            if contrast_mode == 'one':
                anchor_feature = sample_features[:, 0]
                anchor_count = 1
            elif contrast_mode == 'all':
                anchor_feature = contrast_feature
                anchor_count = contrast_count
            else:
                raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T),
                temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # tile mask
            mask = mask.repeat(anchor_count, contrast_count)
            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size * anchor_count).view(-1, 1).to(DEVICE),
                0
            )
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            mask_pos_pairs = mask.sum(1)
            mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

            # loss
            loss = - (temperature / base_temperature) * mean_log_prob_pos
            loss = loss.view(anchor_count, batch_size).mean()

            return loss     

        # print('text_features.shape:{}, \
        #       acoustic_features.shape:{}'.format(text_features.shape,acoustic_features.shape))
        if label is None:
            loss, logits_per_acoustic, logits_per_text = contrastive_loss(acoustic_features,text_features)
        else:
            loss = sup_contrastive_loss(text_features, acoustic_features, labels=label)
        

        if self.option=='match' and self.dataset_name=='gendata':

            paired_feature = torch.cat([text_features, acoustic_features],dim=-1)
            # print('paired_feature.shape:{}'.format(paired_feature.shape))
            match_pred = self.match_discriminator(paired_feature)
            ###match_pred=>(bs, 2)
            match_loss = self.ce_loss(match_pred, category)
            loss = loss + match_loss
        # category_pred = self.haptic_discriminator(paired_feature)
        
        # category_loss = self.category_loss(category_pred, label)

        # loss = cl_loss + category_loss
        #         # print('total_loss:{},cl_loss:{},category:{}'.format(loss, cl_loss, category_loss))

        if mode == 'train':
            return loss
        elif mode == 'valid':
            text_probs = (100.0 * acoustic_features @ text_features.T).softmax(dim=-1)
            # print('text_probs.shape:{}'.format(text_probs.shape))
            tpk = min(text_probs.shape[0], 10)
            top_probs, top_labels = text_probs.cpu().topk(tpk, dim=-1)
            # print('top_probs:{},top_labels:{}'.format(top_probs, top_labels))
            return loss, text_probs, top_probs, top_labels
        else:
            text_probs = (100.0 * acoustic_features @ text_features.T).softmax(dim=-1)
            # print('text_probs.shape:{}'.format(text_probs.shape))
            tpk = min(text_probs.shape[0], 10)
            top_probs, top_labels = text_probs.cpu().topk(tpk, dim=-1)
            # return top_probs,top_labels
            return acoustic_features, text_features, text_probs, top_probs, top_labels
        
        


