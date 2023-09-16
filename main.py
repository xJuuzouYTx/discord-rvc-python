# bot.py
import os
import discord
from dotenv import load_dotenv
from discord.ext.commands import Bot
from inference import Inference
from utils import Audio
import uuid
import typing
from discord import app_commands

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

bot = Bot(command_prefix=".", intents=discord.Intents.default())

def delete_files(paths):
    for path in paths:
        if os.path.exists(path):
            os.remove(path)

models = {}
for root, _, files in os.walk("./weights"):
    pth = None
    index = None

    for file in files:
        if file.endswith(".pth"):
            pth = file
        elif file.endswith(".index"):
            index = file

    if pth or index:
        folder_name = os.path.basename(root)
        models[folder_name] = {
            'pth': os.path.join(folder_name, pth),
            'index': os.path.join("./weights/", folder_name, index) if index else "",
        }

@bot.event
async def on_ready():
    print(f"{bot.user} has connected to Discord!")
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(e)

# @infer.autocomplete("voice")


async def model_autocompletion(
    interaction: discord.Interaction,
    current: str
) -> typing.List[app_commands.Choice[str]]:

    selected_models = []
    for model in models.keys():
        if current.lower() in model.lower():
            selected_models.append(
                app_commands.Choice(name=model, value=model))
    return selected_models


async def method_autocompletion(
    interaction: discord.Interaction,
    current: str
) -> typing.List[app_commands.Choice[str]]:

    selected_methods = []
    for method in ['pm', 'harvest', 'crepe', 'crepe-tiny', 'mangio-crepe', 'mangio-crepe-tiny', 'rmvpe']:
        if current.lower() in method.lower():
            selected_methods.append(
                app_commands.Choice(name=method, value=method))
    return selected_methods


@bot.tree.command(name="infer")
@app_commands.autocomplete(voice=model_autocompletion, method=method_autocompletion)
async def infer(interaction: discord.Interaction, audiofile: discord.Attachment, voice: str, method: str):

    inference = Inference(
        models[voice]['pth'],
        feature_index_path=models[voice]['index'],
        f0_method=method,
    )

    embed = discord.Embed(
        title=f"Cola de Inferencia #{inference.id}",
        description=f"Hey {interaction.user.mention}, estoy procesando tu audio, espera un momento...",
        color=discord.Color.brand_red()
    )
    embed.add_field(name="Nombre del archivo",
                    value=audiofile.filename, inline=False)
    embed.add_field(name="Modelo", value=voice, inline=False)
    embed.add_field(name="Algoritmo", value=method, inline=False)
    await interaction.response.send_message(embed=embed, ephemeral=True)

    uuid_filename = uuid.uuid4()
    file_extension = audiofile.filename.split('.')[-1]
    audio_path = Audio.dowload_from_url(
        url=audiofile.proxy_url, output=f"./audios/{uuid_filename}.{file_extension}")
    inference.source_audio_path = audio_path

    output = await bot.loop.run_in_executor(None, inference.run)

    if output['success']:
        attachment = discord.File(output['file'])
        attachment.filename = audiofile.filename

        await interaction.channel.send(
            content=f"Hey {interaction.user.mention}, ¡tu audio está listo!", 
            file=attachment
        )
    
    delete_files([audio_path, output['file']])
        
bot.run(TOKEN)
