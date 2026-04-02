# URBAN-SYN Dev Setup

This project uses:
- **VS Code Dev Containers** for environment setup (**requires** docker installed)
- **uv** for Python dependency management

---

## 🚀 Quick Start (VS Code Dev Containers)

1. Open repo in VS Code  
2. Press `Ctrl+Shift+P` → **Reopen in Container**  
3. Wait for setup to finish  

That’s it ✅

---

## What happens automatically

- Dev container builds environment
- `uv sync --frozen` runs
- `.venv/` is created with all dependencies

---

## Run Python scripts

```bash
uv run python your_script.py
```

## Notes
- .venv/ is auto-created (do not commit it)
- Always use uv run instead of activating manually
- Keep dependencies reproducible via uv

## Baselines

This project includes baseline implementations adapted from existing research repositories.

### LSTM-TrajGAN
- Source: https://github.com/GeoDS/LSTM-TrajGAN
- Paper: [LSTM-TrajGAN: A Deep Learning Approach to Trajectory Privacy Protection](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.GIScience.2021.I.12)

The code for this baseline is included under:
`baselines/gan/lstm_trajgan/`


See `baselines/gan/lstm_trajgan/NOTES.md` for details.