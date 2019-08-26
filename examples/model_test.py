import torch

from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel

if __name__ == '__main__':
    tokenizer = OpenAIGPTTokenizer.from_pretrained('/home/work/waka/projects/pytorch-pretrained-BERT/samples/LM/Ads/models/epoch1')
    model = OpenAIGPTLMHeadModel.from_pretrained('/home/work/waka/projects/pytorch-pretrained-BERT/samples/LM/Ads/models/epoch1')
    device = torch.device("cuda")
    model.to(device)
    txt = '[BOA] Locked Out? Call Us Now [SEP] Get The Facts You Need To Know [SEP] Fast, Reliable Auto Lockout Services. Call now to Schedule a Service! [EOA]'
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(txt))
    model.eval()
    with torch.no_grad():
        ppls = model.forward_ppl(torch.tensor([ids]).to(device), torch.tensor([len(ids)]).to(device))
    print(ppls.numpy().item())
    pass