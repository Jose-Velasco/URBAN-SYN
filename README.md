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