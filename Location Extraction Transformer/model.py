from torch import nn
from transformers import BertModel


class LET(nn.Module):
    def __init__(self):
        super(LET, self).__init__()

        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True, )

        # Обучаемый слой
        self.linear = nn.Linear(768, 1)

    def forward(self, input_ids, input_mask=None, token_type_ids=None):
        # Выход BERT
        output = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)

        output = output[0]

        # Применение слоя
        y_pred = self.linear(output)
        return y_pred