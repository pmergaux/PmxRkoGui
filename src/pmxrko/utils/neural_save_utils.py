import torch
import os
from datetime import datetime

# =============================================
# SAUVEGARDE D'UN MODÈLE LSTM (dans run_backtest ou evaluate_config)
# =============================================
def save_lstm_model(model, config, score, save_dir="models/simple_opt"):
    os.makedirs(save_dir, exist_ok=True)

    # Nom de fichier intelligent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lstm_best_{score:.5f}_renko{config['renko_size']:.2f}_tb{config['threshold_buy']:.3f}_{timestamp}.pth"
    filepath = os.path.join(save_dir, filename)

    # Ce qu'on sauvegarde (meilleure pratique)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'score': score,
        'timestamp': timestamp,
        'renko_size': config['renko_size'],
        'threshold_buy': config['threshold_buy'],
        'target_col': config.get('target_col', 'unknown')
    }, filepath)

    print(f"MEILLEUR MODÈLE SAUVEGARDÉ → {filepath}")
    return filepath


# =============================================
# RECHARGER LE MEILLEUR MODÈLE À LA FIN
# =============================================
def load_best_lstm_model(model_class, model_path):
    if not os.path.exists(model_path):
        print("Modèle non trouvé !")
        return None

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Recrée le modèle avec la même architecture
    model = model_class(**checkpoint['config'].get('model_params', {}))  # adapte selon ta classe
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Modèle rechargé → score: {checkpoint['score']:.5f}")
    return model, checkpoint

def save_tft_model(model, config, score, model_path):
    torch.save(model, "models/tft_best_2026_COMPLETE.pth")
    print("TFT sauvegardé en entier → méthode 2026")


def load_tft_model(model_class, model_path):
    pass