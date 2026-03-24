# src/inference/translator.py
import torch
import torch.nn.functional as F
from src.model.masks import create_padding_mask, create_decoder_mask

class Translator:
    def __init__(self, model, tokenizer_src, tokenizer_tgt, device):
        self.model = model.to(device)
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.device = device
        self.sos_id = tokenizer_tgt.token_to_id("[SOS]")
        self.eos_id = tokenizer_tgt.token_to_id("[EOS]")
        self.pad_id = tokenizer_tgt.token_to_id("[PAD]")

    def translate(self, sentence, beam_size=3, max_len=50):
        self.model.eval()
        
        #Prepare Source
        src_tokens = self.tokenizer_src.encode(sentence).ids
        src = torch.tensor([src_tokens], device=self.device)
        src_mask = create_padding_mask(src, self.pad_id)

        with torch.no_grad():
            # Pass through Encoder once
            # Note: Encoder now returns (out, weights), we only need out
            enc_out, _ = self.model.encoder(
                self.model.pos_encoder(self.model.src_embedding(src)), 
                src_mask
            )

        # Initialize Beams: (sequence, score, last_attention_weights)
        beams = [(torch.tensor([[self.sos_id]], device=self.device), 0.0, None)]

        for _ in range(max_len):
            candidates = []
            for seq, score, _ in beams:
                # If finished, keep it
                if seq[0, -1].item() == self.eos_id:
                    candidates.append((seq, score, _))
                    continue

                tgt_mask = create_decoder_mask(seq, self.pad_id)
                with torch.no_grad():
                    # Decoder now returns (out, weights)
                    dec_out, attn_weights = self.model.decoder(
                        self.model.pos_encoder(self.model.tgt_embedding(seq)), 
                        enc_out, tgt_mask, src_mask
                    )
                    logits = self.model.fc_out(dec_out[:, -1, :])
                    log_probs = F.log_softmax(logits, dim=-1)

                top_p, top_i = log_probs.topk(beam_size)
                for i in range(beam_size):
                    next_token = top_i[:, i:i+1]
                    next_score = score + top_p[0, i].item()
                    candidates.append((torch.cat([seq, next_token], dim=1), next_score, attn_weights))

            # Sort and prune
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            
            # Stop if all beams found EOS
            if all(b[0][0, -1].item() == self.eos_id for b in beams): 
                break

        # Best sequence and its cross-attention weights
        best_seq, _, best_attn = beams[0]
        
        # Convert IDs to tokens for visualizations
        src_labels = self.tokenizer_src.encode(sentence).tokens
        tgt_labels = self.tokenizer_tgt.decode_batch([best_seq.squeeze().tolist()])[0].split()
        # Add [SOS] and [EOS] labels for the target axis if they exist in the tensor
        tgt_labels = ["[SOS]"] + tgt_labels + ["[EOS]"]

        translated_text = self.tokenizer_tgt.decode(best_seq.squeeze().tolist())
        
        return translated_text, best_attn, src_labels, tgt_labels