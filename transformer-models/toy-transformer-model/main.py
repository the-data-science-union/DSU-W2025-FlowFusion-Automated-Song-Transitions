from transformers import BertTokenizer
from utils.data_utils import get_nltk_dataloader
from model.model import BERT
from train import BERTTrainer
import config

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataloader = get_nltk_dataloader(tokenizer, config.MAX_LENGTH, config.BATCH_SIZE)

    model = BERT(config.VOCAB_SIZE, config.D_MODEL, config.NUM_LAYERS, config.NUM_HEADS, config.D_FF, config.MAX_LENGTH, config.DROPOUT)
    trainer = BERTTrainer(model, config.VOCAB_SIZE, config.MAX_LENGTH, config.LEARNING_RATE, config.WARMUP_STEPS)

    trainer.train(dataloader, config.NUM_EPOCHS)
    trainer.save_model(os.path.join(config.OUTPUT_DIR, "bert_model.pth"))

if __name__ == "__main__":
    main()