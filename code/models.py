import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN
from einops import reduce
from network.tools import FeedForward,MultiHeadLinearAttention,SelfAttention
from network.guided_cross_attention_model import GuidedCrossAttention

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss

def mean_square_error(v_d, v_s):
    loss_fct = torch.nn.MSELoss()
    loss = loss_fct(v_d, v_s)
    return loss

def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super(EncoderLayer, self).__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout, device)
        self.ea = SelfAttention(hid_dim, n_heads, dropout, device)
        self.ff = FeedForward(hid_dim, hid_dim, glu=True, dropout=dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, trg_mask=None, src_mask=None):
        #self attetion
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        #feed forward
        trg = self.ln(trg + self.do(self.ff(trg)))
        return trg

class GuideGateDTI(nn.Module):
    def __init__(self, device,**config):
        super(GuideGateDTI, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        drug_padding = config["DRUG"]["PADDING"]
        atom_dim = config["DRUG"]["ATOM_DIM"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        protein_num_head = config['PROTEIN']['NUM_HEAD']
        bert_embedding = config["PROTEIN"]["BERT_EMBEDDING_DIM"]
        hid_dim = config["MGN"]["EMBEDDING_DIM"]
        tgca_num_head = config["SOLVER"]["tgca_num_head"]
        dropout = config["SOLVER"]["DROPOUT"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        out_binary = config["DECODER"]["BINARY"]
        use_pcsa = config["DA"]["TGSA"]
        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)

        self.protein_extractor = EncoderLayer(protein_emb_dim, protein_num_head, dropout,device)

        #Guided Gating Network
        self.Guided_gating_network = GuidedgatingNetwork(hid_dim,tgca_num_head,use_pcsa)
        #MLPDecoder
        self.mlp_classifier = MLPDecoder(mlp_in_dim*2, mlp_hidden_dim*2, mlp_out_dim*2, binary=out_binary)
        self.ft = nn.Linear(atom_dim, drug_embedding)
        self.fc = nn.Linear(bert_embedding, protein_emb_dim)
        
    def forward(self, smi_d, bg_d, v_p,mode="train"):
        #Drug Encoder
        v_d = self.drug_extractor(bg_d)
        v_s = self.ft(smi_d)
        #Protein Encoder
        v_p = self.fc(v_p)
        v_p = self.protein_extractor(v_p)
        #Guided Gateing Network
        f, v_d, v_s, v_p = self.Guided_gating_network(v_d, v_s, v_p)
        score = self.mlp_classifier(f)
        
        if mode == "train":
            return v_d, v_s, v_p, f, score
        elif mode == "eval":
            return v_d, v_s, v_p, f,score


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats
class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v


class GLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GLU, self).__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.V = nn.Linear(in_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        Y = self.W(X) * self.sigmoid(self.V(X))
        return Y
        
class TGSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
    def forward(self, x, cond_feat):  # x: [B, L, D], cond_feat: [B, D]
        B, L, D = x.shape
        # Step 1: Conditional gating vector
        gate = torch.sigmoid(cond_feat).unsqueeze(1)  # [B, 1, D]

        # Step 2: Project features
        K = self.k_proj(x) * gate                     # gated key
        Q = self.q_proj(x) * gate                     # gated query
        V = self.v_proj(x)                            # value remains original
        # Step 3: Attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)  # [B, L, L]
        attn_weights = torch.softmax(attn_scores, dim=-1)                # [B, L, L]
        out = torch.matmul(attn_weights, V)                              # [B, L, D]
        # Step 4: Final projection
        return self.out_proj(out)                                        # [B, L, D]
    
class GuidedgatingNetwork(nn.Module):
    def __init__(self,dim,n_heads,use_pcsa):
        super(GuidedgatingNetwork, self).__init__()
        self.gated_g = GLU(dim*2, dim*2)
        self.gated_s = GLU(dim*2, dim*2)
        self.gated_p = GLU(dim, dim)
        self.coa_dp = GuidedCrossAttention(embed_dim=dim, num_heads=n_heads)
        self.v_mhla = MultiHeadLinearAttention(d_model=dim * 2, d_diff=dim* 8, nhead=8, dropout=0, activation='gelu')
        self.v_gca_norm = nn.LayerNorm(dim * 2)
        self.coa_sp = GuidedCrossAttention(embed_dim=dim, num_heads=n_heads)
        self.x_mhla = MultiHeadLinearAttention(d_model=dim * 2, d_diff=dim * 8, nhead=8, dropout=0, activation='gelu')
        self.x_gca_norm = nn.LayerNorm(dim * 2)
        self.tanh = nn.Tanh()
        self.tgsa_d = TGSA(dim=dim)
        self.tgsa_s = TGSA(dim=dim)
        self.use_pcsa = use_pcsa  

    def forward(self, mg, ms, mp,trg_mask=None,src_mask=None):
        # ========== Step 1: Transpose for MultiheadAttention input ==========
        mg_t = mg.transpose(0, 1)  # [290, B, 128]
        ms_t = ms.transpose(0, 1)  # [24, B, 128]
        mp_t = mp.transpose(0, 1)  # [1527, B, 128]
        # ========== Step 2: Extract protein summary vector ==========
        # mean pooling
        mean_pool = torch.mean(mp, dim=1)        
        # max pooling
        max_pool = torch.max(mp, dim=1)[0] 

        mp_summary = mean_pool + max_pool        
        # ========== Step 3: PCSA (Protein-Conditioned Self-Attention) ==========
        mg_input = mg_t.transpose(0, 1)  
        ms_input = ms_t.transpose(0, 1)  

        if self.use_pcsa:
            mg_pcsa = self.tgsa_d(mg_input, mp_summary)  
            ms_pcsa = self.tgsa_s(ms_input, mp_summary)  
        else:
            mg_pcsa = mg_input  
            ms_pcsa = ms_input
        mg_pcsa_t = mg_pcsa.transpose(0, 1)  
        ms_pcsa_t = ms_pcsa.transpose(0, 1)  
        attn_dp, _ = self.coa_dp(mg_pcsa_t, mp_t, mp_t)  
        attn_sp, _ = self.coa_sp(ms_pcsa_t, mp_t, mp_t)  
        attn_dp = attn_dp.transpose(0, 1) 
        attn_sp = attn_sp.transpose(0, 1) 
        attn_dp = torch.cat((mg_pcsa_t.transpose(0, 1), attn_dp), 2)
        attn_sp = torch.cat((ms_pcsa_t.transpose(0, 1), attn_sp), 2)
        dp = attn_dp
        sp = attn_sp
        attn_dp = self.v_mhla(dp)
        attn_sp = self.x_mhla(sp)
        attn_dp = attn_dp + dp
        attn_sp = attn_sp + sp
        attn_dp = self.v_gca_norm(attn_dp)
        attn_sp = self.x_gca_norm(attn_sp)
        # ========== Step 5: Gated Projection ==========
        v_dp = self.gated_g(attn_dp)  
        v_sp = self.gated_s(attn_sp)  

        # ========== Step 6: Max Pooling ==========
        v_dp = torch.max(v_dp, dim=1).values  
        v_sp = torch.max(v_sp, dim=1).values  

        # ========== Step 7: Optional raw modality pooling (供分析或对比用) ==========
        v_d = reduce(mg, "b h w -> b w", 'max')  
        v_s = reduce(ms, "b h w -> b w", 'max')  
        v_p = reduce(mp, "b h w -> b w", 'max')  
        # ========== Step 8: Final feature fusion ==========
        f = self.tanh(torch.cat([v_dp, v_sp], dim=-1))  
        return f, v_dp, v_sp, v_p

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.act3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)


    def forward(self, x):#x.shpae[128,512]
        x = self.bn1(self.act1(self.fc1(x)))
        x = self.bn2(self.act2(self.fc2(x)))
        x = self.bn3(self.act3(self.fc3(x)))
        x = self.fc4(x)
        return x
    