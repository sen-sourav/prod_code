import discord
import requests
import os

TOKEN = 'your_discord_bot_token'
client = discord.Client()

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('!process'):
        if len(message.attachments) > 0:
            attachment = message.attachments[0]
            input_file_path = f"uploads/{attachment.filename}"
            os.makedirs(os.path.dirname(input_file_path), exist_ok=True)
            await attachment.save(input_file_path)
            
            files = {'file': open(input_file_path, 'rb')}
            response = requests.post('http://your_server_ip:8000/inference', files=files)
            
            if response.status_code == 200:
                response_data = response.json()
                if 'output_file' in response_data:
                    output_file_path = response_data['output_file']
                    await message.channel.send(file=discord.File(output_file_path))
                else:
                    await message.channel.send("There was an error processing your file.")
            else:
                await message.channel.send("There was an error processing your file.")

def run_discord_bot():
    client.run(TOKEN)
