import os
import discord
from dotenv import load_dotenv
from discord.ext.commands import Bot
from inference import Inference
from myutils import Audio
import uuid
import typing
from discord import app_commands
from myutils import delete_files

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

bot = Bot(command_prefix=".", intents=discord.Intents.default())

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


class InferMessage:
    def __init__(self, interaction: discord.Interaction, audiofile: discord.Attachment, voice: str, method: str) -> None:
        self.interaction = interaction
        self.audiofile = audiofile
        self.voice = voice
        self.method = method


class AudioQueue:
    def __init__(self) -> None:
        self.queue = []
        self.processing = False

    async def push(self, interaction: discord.Interaction, audiofile: discord.Attachment, voice: str, method: str):
        # Check if has license
        user = True

        if not user:
            embed = discord.Embed(
                title=f"¡Lo sentimos!",
                description=f"No tienes una suscripción activa, por favor visita https://voni.vercel.app para obtener una suscripción",
                color=discord.Color.brand_red()
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
        else:
            embed = discord.Embed(
                title=f"Cola de Inferencia #{len(self.queue)}",
                description=f"Hey {interaction.user.mention}, estoy procesando tu audio, espera un momento...",
                color=discord.Color.brand_red()
            )
            embed.add_field(name="Nombre del archivo",
                            value=audiofile.filename, inline=False)
            embed.add_field(name="Algoritmo", value=method, inline=False)
            await interaction.response.send_message(embed=embed)
            
                
            infer_message = InferMessage(interaction, audiofile, voice, method)
            self.queue.append(infer_message)

            if not self.processing:
                await self.process()

    async def process(self):
        self.processing = True

        for current in self.queue:
            uuid_filename = uuid.uuid4()
            file_extension = current.audiofile.filename.split('.')[-1]
            audio_path = Audio.dowload_from_url(
                url=current.audiofile.proxy_url,
                output=f"./audios/{uuid_filename}.{file_extension}"
            )
            inference = Inference(
                f0_method=current.method,
                model_name=current.voice
            )

            inference.source_audio_path = audio_path

            output = await bot.loop.run_in_executor(None, inference.run)

            if 'success' in output and output['success']:
                attachment = discord.File(output['file'])
                attachment.filename = current.audiofile.filename

                await current.interaction.followup.send(
                    content=f"Hey {current.interaction.user.mention}, ¡tu audio está listo!",
                    file=attachment
                )
            else:
                embed = discord.Embed(
                    title=f"¡Lo sentimos!",
                    description="Ocurrió un error al convertir tu archivo",
                    color=discord.Color.brand_red()
                )
                await current.interaction.followup.send(embed=embed, ephemeral=True)

            delete_files([audio_path, f"{audio_path}.wav", output['file']])
            self.queue.remove(current)
            self.processing = False


audioQueue = AudioQueue()

@bot.event
async def on_ready():
    print(f"{bot.user} has connected to Discord!")
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(e)


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
@app_commands.autocomplete(method=method_autocompletion)
async def infer(interaction: discord.Interaction, audiofile: discord.Attachment, voice: str, method: str):
    await audioQueue.push(interaction, audiofile, voice, method)

bot.run(TOKEN)
