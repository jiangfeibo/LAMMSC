import copy
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import random
import numpy as np
import json
from channel_nets import channel_net
import os
class params():
    checkpoint_path = "checkpoints" # path to model weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = r"E:\datasets\VOC2012_img2text" # path to your text data
    log_path = "logs" # path to logs
    epoch = 20 # training epoch
    lr = 1e-3 # learning rate
    batchsize = 16 # training batch size
    snr = 15 # SNR
    weight_delay = 1e-5
    sim_th = 0.6 # cosine similarity threshold
    emb_dim = 768 # embedding size of the SC model
    n_heads = 8 # head number of the SC model
    hidden_dim = 1024 # dimensions of hidden layers in the SC model
    num_layers = 2 # layers of the transformer decoder in the SC model
    use_CGE = True # use CGE for channel estimation or not
    max_length = 30 # maximum number of words in each text

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Transformer-based text SC model
class TextSCNet(nn.Module):
    def __init__(self, emb_dim, n_heads, hidden_dim, num_layers):
        super(TextSCNet, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased') # use Bert as the Semantic encoder
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(emb_dim, n_heads, hidden_dim), num_layers)
        self.fc = nn.Linear(emb_dim, self.encoder.config.vocab_size)
        self.channel_model = channel_net(in_dims=23040, snr=arg.snr,CGE=arg.use_CGE)
        self.embedding = nn.Embedding(self.encoder.config.vocab_size, emb_dim)
    def forward(self, src_input_ids, src_attention_mask, trg_input_ids):
        s_code = self.encoder(src_input_ids, attention_mask=src_attention_mask).last_hidden_state
        b_s,w_s,f_s = s_code.shape
        s_code = s_code.view(b_s,-1)
        ch_code, ch_code_, s_code_d = self.channel_model(s_code) # transmit on channel
        trg_emb = self.embedding(trg_input_ids)
        s_code_ = s_code_d.view(b_s,w_s,f_s)
        decoded = self.decoder(trg_emb, s_code_)
        decoded_output = self.fc(decoded)
        return ch_code, ch_code_, s_code, s_code_d, decoded_output

# SC model training
def SC_train(model,tokenizer, training_texts, arg):
    model.to(arg.device)
    criterion = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    weights_path = f"{arg.checkpoint_path}/TextSC_snr{arg.snr}.pth"
    raw_text = []
    rec_text = []
    for epoch in range(arg.epoch):
        random.shuffle(training_texts)
        model.train()
        for i in range(0,len(training_texts),arg.batchsize):
            if i+arg.batchsize < len(training_texts):
                b_text = training_texts[i:i+arg.batchsize]
            else:
                break
            raw_text += b_text
            optimizer.zero_grad()
            input_text = b_text
            encoded_dict = tokenizer.batch_encode_plus(
                input_text,  # Input sentence
                add_special_tokens=True,  # Add special tokens, such as [CLS] and [SEP]
                max_length=arg.max_length,  # Set the maximum length, truncate if exceeded
                pad_to_max_length=True,  # Pad to the maximum length
                return_attention_mask=True,  # Return attention masks
                return_tensors='pt'  # Return tensor type, here is PyTorch
            )
            # Extract input ids, attention masks and token type ids from the dictionary
            input_ids = encoded_dict['input_ids'].to(arg.device)
            encoded_dict_d = tokenizer.batch_encode_plus(
                input_text,  # Input sentence
                add_special_tokens=True,  # Add special tokens, such as [CLS] and [SEP]
                max_length=arg.max_length,  # Set the maximum length, truncate if exceeded
                pad_to_max_length=True,  # Pad to the maximum length
                return_attention_mask=True,  # Return attention masks
                return_tensors='pt'  # Return tensor type, here is PyTorch
            )
            target_ids = encoded_dict_d['input_ids'].to(arg.device)

            src_input_ids = input_ids.clone()
            trg_input_ids = target_ids.clone()
            src_attention_mask = (src_input_ids != tokenizer.pad_token_id).float().to(arg.device)
            ch_code, ch_code_, s_code, s_code_, output = model(src_input_ids, src_attention_mask, trg_input_ids)

            loss_ch = mse(s_code,s_code_)

            loss_SC = criterion(output.view(-1, model.encoder.config.vocab_size),
                             target_ids.contiguous().view(-1))

            loss = loss_SC + loss_ch
            loss.backward()
            optimizer.step()
            ## recover the text
            for i,o in enumerate(output):
                predicted_indices = torch.argmax(o.view(-1, model.encoder.config.vocab_size), dim=1).cpu().numpy()
                predicted_sentence = tokenizer.decode(predicted_indices, skip_special_tokens=True)
                print("src:",input_text[i], '\nrec:', predicted_sentence)
                rec_text.append(predicted_sentence)
            with open(os.path.join(arg.log_path,f"t2t_snr{arg.snr}_res.json"),"w",encoding="utf-8")as f:
                f.write(json.dumps({"raw_text":raw_text,"rec_text":rec_text},indent=4,ensure_ascii=False))
            print(f"epoch {epoch}, loss: {loss.item()}")
        torch.save(model.state_dict(), weights_path)

# Data transmission based on the trained SC model
@torch.no_grad()
def data_transmission(input_text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    SC_model = TextSCNet(arg.emb_dim, arg.n_heads, arg.hidden_dim, arg.num_layers).to(arg.device)
    weight = torch.load(f"{arg.checkpoint_path}/TextSC_snr{arg.snr}.pth",map_location="cpu")
    SC_model.load_state_dict(weight)
    SC_model.eval()
    encoded_dict = tokenizer.batch_encode_plus(
        input_text,  # Input sentence
        add_special_tokens=True,  # Add special tokens, such as [CLS] and [SEP]
        max_length=arg.max_length,  # Set the maximum length, truncate if exceeded
        pad_to_max_length=True,  # Pad to the maximum length
        return_attention_mask=True,  # Return attention masks
        return_tensors='pt'  # Return tensor type, here is PyTorch
    )
    # Extract input ids, attention masks and token type ids from the dictionary
    input_ids = encoded_dict['input_ids'].to(arg.device)
    src_input_ids = input_ids.clone()
    src_attention_mask = (src_input_ids != tokenizer.pad_token_id).float().to(arg.device)
    encoded_dict_d = tokenizer.batch_encode_plus(
        input_text,  # Input sentence
        add_special_tokens=True,  # Add special tokens, such as [CLS] and [SEP]
        max_length=arg.max_length,  # Set the maximum length, truncate if exceeded
        pad_to_max_length=True,  # Pad to the maximum length
        return_attention_mask=True,  # Return attention masks
        return_tensors='pt'  # Return tensor type, here is PyTorch
    )
    trg_input_ids = encoded_dict_d['input_ids'].to(arg.device)
    ch_code, ch_code_, s_code, s_code_, output = SC_model(src_input_ids, src_attention_mask, trg_input_ids)
    rec_text = []
    for o in output:
        predicted_indices = torch.argmax(o.view(-1, SC_model.encoder.config.vocab_size), dim=1).cpu().numpy()
        predicted_sentence = tokenizer.decode(predicted_indices, skip_special_tokens=True)
        rec_text.append(predicted_sentence)
    print(rec_text)
    return rec_text

arg = params()
if __name__ == '__main__':
    same_seeds(1024)
    train_data = []
    for text in os.listdir(arg.dataset):
        if text.endswith(".json"):
            text_path = os.path.join(arg.dataset, text)
            with open(text_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                content = [val.replace("<unk>", "") for val in content]
            train_data += content
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    SC_model = TextSCNet(arg.emb_dim, arg.n_heads, arg.hidden_dim, arg.num_layers).to(arg.device)
    SC_train(SC_model,tokenizer,train_data,arg)