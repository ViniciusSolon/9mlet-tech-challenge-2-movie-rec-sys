# SetAI Configuration

This folder contains the SetAI CLI configuration used to generate this project structure.

## Contents

- `config.json` - CLI configuration (API keys, language settings, etc.)
- `.gitignore` - Prevents committing sensitive configuration to version control

## Important Notes

- **⚠️ DO NOT commit this folder** - It contains sensitive information like API keys
- **⚠️ SECURITY WARNING:** This folder contains real API keys. Never commit it to version control
- The `.gitignore` file is included to prevent accidental commits, but always verify before pushing
- To update configuration, use `setai config` command in your terminal

## Configuration Location

The actual configuration file with full API keys is located at:
- **Windows:** `%USERPROFILE%\.setai\config.json`
- **macOS/Linux:** `~/.setai/config.json`

This folder (`.setai`) is a reference copy that shows what configuration was used when generating the project structure.
