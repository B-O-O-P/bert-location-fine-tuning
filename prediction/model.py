from transformers import BertModel
from torch import bmm, nn

class BERT(nn.Module):
    def __init__(self, embedding=None):
        super(BERT, self).__init__()
        # Изначальная модель
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, )
        # Эмбеддинг
        self.embedding = embedding
        # Линейный слой
        self.linear = nn.Linear(768, 256)

    def forward(self, input_ids, input_background, input_mask=None, token_type_ids=None):
        # Получение cls токена
        output = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        output = output['pooler_output']
        # Применение линейного слоя
        output = self.linear(output).unsqueeze(1)

        # Получение эмбеддинга для бэкграунда
        background = self.embedding(input_background).unsqueeze(2)

        # Скалярное произведение
        y_pred = bmm(output, background)
        return y_pred
