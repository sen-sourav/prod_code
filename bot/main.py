import bot
from app.model_loader import load_model

if __name__ == '__main__':
    model = load_model()
    bot.run_discord_bot(model=model)
