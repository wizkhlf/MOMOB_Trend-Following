# MAMOB-V2-Trend-Following

Backtest **trend following** multi-actifs (univers CAC40-like) avec comparaison **1-couple** vs **3-couples** de moyennes mobiles, sélection **walk-forward** (70% IS / 30% OOS) et diagnostics de robustesse.

## Contenu
- `app.py` : moteur de backtest + routes Flask (UI + diagnostics)
- `main.py` : lance l’interface web
- `templates/` : pages HTML (résultats + diagnostics)
- `notebooks/Methodology.ipynb` : notebook expliquant la démarche (V1 → V2), difficultés et choix

## Installation
```bash
pip install -r requirements.txt
```

## Lancer l'interface web
```bash
python main.py
```
Puis ouvrir : `http://127.0.0.1:5002`

Alternative: Lancer l'applcation en ligne via: https://momob-trend-following.onrender.com/

## Notes
- Les données proviennent de **Yahoo Finance** via `yfinance` (qualité variable selon les tickers/périodes).
- Les résultats “OOS” (out-of-sample) correspondent à la partie test après split.
