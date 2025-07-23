import os
import torch
from loader.dataset import E2EDataset
from loader.feature_extractor import FeatureExtractor
from loader.tokenizer import CharTokenizer
from model import E2EDASTModel
from utils.helpers import load_config
from utils.logger import get_logger
from torch.utils.data import DataLoader
from tqdm import tqdm

def infer(config_path='config/config.yaml', checkpoint_path='checkpoints/model_epoch1.pt'):
    config = load_config(config_path)
    logger = get_logger(config['inference']['log_path'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CharTokenizer()
    feature_extractor = FeatureExtractor(config['feature_extractor'])

    test_dataset = E2EDataset(config['data']['test_dir'], tokenizer, feature_extractor)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = E2EDASTModel(config['model'], tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Inference")):
            input_features = batch['input_features'].to(device)
            file_name = batch['file_name'][0]

            logits, diar_logits, _ = model(input_features)
            preds = torch.argmax(logits, dim=-1).squeeze(0)
            pred_text = tokenizer.decode(preds.cpu().numpy())

            diar_labels = torch.argmax(diar_logits, dim=-1).squeeze(0).cpu().numpy()
            output = f"File: {file_name}\nPrediction: {pred_text}\nDiarization: {diar_labels}\n"
            logger.info(output)
            print(output)

if __name__ == "__main__":
    infer()
