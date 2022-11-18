import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
import torch


class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # TODO: fix hard coding of number of labels
        self.num_labels_original = 28
        self.num_labels_group = 4

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_layer = nn.Linear(config.hidden_size, 1024)
        self.act = nn.LeakyReLU()
        self.classifier_original = nn.Linear(1024, self.num_labels_original)
        self.classifier_group = nn.Linear(1024, self.num_labels_group)
        self.loss_fct_1 = nn.BCEWithLogitsLoss()
        self.loss_fct_2 = nn.BCEWithLogitsLoss()

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            sources=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        hidden_state = self.act(self.hidden_layer(pooled_output))
        logits_original = self.classifier_original(hidden_state)
        logits_group = self.classifier_group(hidden_state)

        # TODO: check if it needs device
        original_idx = sources == 0
        group_idx = sources == 1

        # TODO: keep in mind that error can arise in indexing below if original_idx and group_idx above are None
        original_labels = labels[original_idx, :28]
        group_labels = labels[group_idx, 28:]

        original_predictions = logits_original[original_idx, :]
        group_predictions = logits_group[group_idx, :]

        if original_labels.numel() != 0:
            loss_1 = self.loss_fct_1(original_predictions, original_labels)
        else:
            loss_1 = 0

        if group_labels.numel() != 0:
            loss_2 = self.loss_fct_2(group_predictions, group_labels)
        else:
            loss_2 = 0

        loss = loss_1 + loss_2

        outputs = (loss, original_predictions, group_predictions)
        return outputs

        # separate individual task predictions based on the source of the datapoints
        # logits = torch.zeros(sources.size(dim=0), self.num_labels_original + self.num_labels_group, device=device)
        #logits = logits.to(device)

        # for i in range(sources.size(dim=0)):
        #     original = logits_original[i]
        #     original = original.to(device)
        #     original_ext = torch.tensor([0.0]*self.num_labels_group, device=device)
        #     #original_ext = original_ext.to(device)
        #     original_preds = torch.cat((original, original_ext), 0)
        #     #original_preds = original_preds.to(device)
        #
        #     group = logits_group[i]
        #     group = group.to(device)
        #     group_ext = torch.tensor([0.0]*self.num_labels_original, device=device)
        #     #group_ext = group_ext.to(device)
        #     group_preds = torch.cat((group_ext, group), 0)
        #     group_preds = group_preds.to(device)
        #
        #     s_original = torch.LongTensor([0], device=device)
        #     #s_original = s_original.to(device)
        #     s_group = torch.LongTensor([1], device=device)
        #     #s_group = s_group.to(device)
            
            # if sources[i] == s_original:
            #     prediction = original_preds
            # elif sources[i] == s_group:
            #     prediction = original_preds
            # logits[i, :] = prediction.to(device)

        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        #
        # if labels is not None:
        #     loss = self.loss_fct(logits, labels)
        #     outputs = (loss,) + outputs
        #
        # return outputs  # (loss), logits, (hidden_states), (attentions)