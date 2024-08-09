import discord
from discord.ext import commands
import requests
import os
from app.model_inference import model_inference

TOKEN = 'REDACTED'  # Replace with your bot's token


# # Define a subclass of commands.Bot
class MyBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True  # Enable message content intent
        model = None
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self):
        # Sync the commands with the Discord server
        await self.tree.sync()
        print("Commands synced.")


# Initialize the bot
bot = MyBot()

# bot = commands.Bot(command_prefix="!", intents=discord.Intents.default())
bot.tree.sync()


#  Define an event for when the bot is ready
@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')


# Define the /SingSong command
@bot.tree.command(name="singsong")
async def singsong(interaction: discord.Interaction):
    await interaction.response.send_message(
        "Please join a voice channel and sing your song. You can then upload the recording here!")


@bot.event
async def on_message(message):
    # Ignore messages by bot, this prevents infinite loop
    if message.author == bot.user:
        return

    if message.attachments:
        attachment = message.attachments[0]
        if attachment.filename.endswith(('mp3', 'wav', 'ogg', 'm4a')):
            await message.channel.send(f"Received voice recording: {message.author.name}_{attachment.filename}")
            # You can save or process the file here
            filename = f"{message.author.name}_{attachment.filename}"
            await attachment.save(f'/home/ubuntu/prod_code/uploads/{filename}')
            input_file_path = f"/home/ubuntu/prod_code/uploads/{filename}"
            output_dir = "/home/ubuntu/prod_code/output/"
            model_inference(bot.model, input_file_path, output_dir)
            basename = filename.split('.')[-2]
            file = discord.File(
                f'/home/ubuntu/prod_code/output/{basename}_final_mix.wav')  # replace with your file path
            await message.channel.send(file=file)
    await bot.process_commands(message)


def run_discord_bot(model=None):
    bot.model = model
    bot.run(TOKEN)
