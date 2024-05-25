import os
import asyncio
import signal
import sys
import threading
import traceback
from pathlib import Path
from platform import system

import discord

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, UpdateStatus
from qdrant_client.http import models as models

from pycord.multicog import apply_multicog

from cogs.code_interpreter_service_cog import CodeInterpreterService
from cogs.search_service_cog import SearchService
from cogs.text_service_cog import GPT3ComCon
from cogs.image_service_cog import DrawDallEService
from cogs.prompt_optimizer_cog import ImgPromptOptimizer
from cogs.moderations_service_cog import ModerationsService
from cogs.commands import Commands
from cogs.transcription_service_cog import TranscribeService
from cogs.translation_service_cog import TranslationService
from cogs.index_service_cog import IndexService
from models.deepl_model import TranslationModel
from services.health_service import HealthService
from services.pickle_service import Pickler

from services.qdrant_service import QdrantService
from services.deletion_service import Deletion
from services.message_queue_service import Message
from services.usage_service import UsageService
from services.environment_service import EnvService

from models.openai_model import Model


__version__ = "12.3.8"


PID_FILE = Path("bot.pid")
PROCESS = None

if sys.platform == "win32":
    separator = "\\"
else:
    separator = "/"

#
# The qdrant service is used to store and retrieve conversation embeddings.
#

try:
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_HOST = os.getenv("QDRANT_HOST")
    QDRANT_PORT = os.getenv("QDRANT_PORT")
    QDRANT_INDEX_NAME = "gptdiscord_conversation-embeddings"
except Exception:
    QDRANT_API_KEY = None
    
vector_size = 1536
vector_distance = Distance.COSINE

qdrant_service = None
if QDRANT_API_KEY and QDRANT_HOST and QDRANT_PORT and QDRANT_INDEX_NAME:
    QDRANT_URL = QDRANT_HOST + ':' + QDRANT_PORT
    client = QdrantClient(api_key=QDRANT_API_KEY, url=QDRANT_URL)
    try:
        collection = client.get_collection(collection_name=QDRANT_INDEX_NAME)
    except Exception as e:
        if "404" in str(e):
            print("Creating Qdrant index. Please wait...")
            try:
                # Create the collection if it doesn't exist
                client.create_collection(
                    collection_name=QDRANT_INDEX_NAME,
                    vectors_config=VectorParams(size=vector_size, distance=vector_distance)
                )
            except Exception as create_error:
                print(f"Failed to create collection: {str(create_error.content)}")
        else:
            print(f"Unexpected error: {str(e)}")

    qdrant_service = QdrantService(client, collection_name=QDRANT_INDEX_NAME)
    print("Got the Qdrant service")
    
#
# Message queueing for the debug service, defer debug messages to be sent later so we don't hit rate limits.
#
message_queue = asyncio.Queue()
deletion_queue = asyncio.Queue()
asyncio.ensure_future(Message.process_message_queue(message_queue, 1.5, 5))
asyncio.ensure_future(Deletion.process_deletion_queue(deletion_queue, 1, 1))

# Pickling service for conversation persistence
try:
    Path(EnvService.save_path() / "pickles").mkdir(exist_ok=True)
except Exception:
    traceback.print_exc()
    print(
        "Could not start pickle service. Conversation history will not be persistent across restarts."
    )
pickle_queue = asyncio.Queue()
asyncio.ensure_future(Pickler.process_pickle_queue(pickle_queue, 5, 1))


#
# Settings for the bot
#
activity = discord.Activity(
    type=discord.ActivityType.watching, name="for /help, /gpt, and more!"
)
bot = discord.Bot(intents=discord.Intents.all(), command_prefix="!", activity=activity)
usage_service = UsageService(Path(os.environ.get("DATA_DIR", os.getcwd())))
model = Model(usage_service)


#
# An encapsulating wrapper for the discord.py client. This uses the old re-write without cogs, but it gets the job done!
#


@bot.event  # Using self gives u
async def on_ready():  # I can make self optional by
    print("We have logged in as {0.user}".format(bot))


@bot.event
async def on_application_command_error(
    ctx: discord.ApplicationContext, error: discord.DiscordException
):
    if isinstance(error, discord.CheckFailure):
        pass
    else:
        raise error


async def main():
    data_path = EnvService.environment_path_with_fallback("DATA_DIR")
    debug_guild = int(os.getenv("DEBUG_GUILD"))
    debug_channel = int(os.getenv("DEBUG_CHANNEL"))

    if not data_path.exists():
        raise OSError(f"Data path: {data_path} does not exist ... create it?")

    # Load the cog for the moderations service
    bot.add_cog(ModerationsService(bot, usage_service, model))

    # Load the main GPT3 Bot service
    bot.add_cog(
        GPT3ComCon(
            bot,
            usage_service,
            model,
            message_queue,
            deletion_queue,
            debug_guild,
            debug_channel,
            data_path,
            qdrant_service=qdrant_service,
            pickle_queue=pickle_queue,
        )
    )

    bot.add_cog(
        DrawDallEService(
            bot,
            usage_service,
            model,
            message_queue,
            deletion_queue,
            bot.get_cog("GPT3ComCon"),
        )
    )

    bot.add_cog(
        ImgPromptOptimizer(
            bot,
            usage_service,
            model,
            message_queue,
            deletion_queue,
            bot.get_cog("GPT3ComCon"),
            bot.get_cog("DrawDallEService"),
        )
    )

    bot.add_cog(
        IndexService(
            bot,
            usage_service,
            deletion_queue,
        )
    )

    if EnvService.get_deepl_token():
        bot.add_cog(TranslationService(bot, TranslationModel()))
        print("The translation service is enabled.")

    if (
        EnvService.get_google_search_api_key()
        and EnvService.get_google_search_engine_id()
    ):
        bot.add_cog(
            SearchService(
                bot, model, usage_service, deletion_queue, bot.get_cog("GPT3ComCon")
            )
        )
        print("The Search service is enabled.")

    if EnvService.get_e2b_api_key():
        bot.add_cog(
            CodeInterpreterService(
                bot, model, usage_service, deletion_queue, bot.get_cog("GPT3ComCon")
            )
        )
        print("The Code Interpreter service is enabled.")

    bot.add_cog(
        TranscribeService(
            bot,
            model,
            usage_service,
        )
    )

    bot.add_cog(
        Commands(
            bot,
            usage_service,
            model,
            message_queue,
            deletion_queue,
            bot.get_cog("GPT3ComCon"),
            bot.get_cog("DrawDallEService"),
            bot.get_cog("ImgPromptOptimizer"),
            bot.get_cog("ModerationsService"),
            bot.get_cog("IndexService"),
            bot.get_cog("TranslationService"),
            bot.get_cog("SearchService"),
            bot.get_cog("TranscribeService"),
            bot.get_cog("CodeInterpreterService"),
        )
    )

    apply_multicog(bot)

    await bot.start(os.getenv("DISCORD_TOKEN"))


def check_process_file(pid_file: Path) -> bool:
    """Check the pid file exists and if the Process ID is actually running"""
    if not pid_file.exists():
        return False
    if system() == "Linux":
        with pid_file.open("r") as pfp:
            try:
                proc_pid_path = Path("/proc") / "{int(pfp.read().strip())}"
                print("Checking if PID proc path {proc_pid_path} exists")
            except ValueError:
                # We don't have a valid int in the PID File^M
                pid_file.unlink()
                return False
        return proc_pid_path.exists()
    return True


def cleanup_pid_file(signum, frame):
    # Kill all threads
    if PROCESS:
        print("Killing all subprocesses")
        PROCESS.terminate()
    print("Killed all subprocesses")
    # Always cleanup PID File if it exists
    if PID_FILE.exists():
        print(f"Removing PID file {PID_FILE}", flush=True)
        PID_FILE.unlink()


# Run the bot with a token taken from an environment file.
def init():
    global PROCESS
    # Handle SIGTERM cleanly - Docker sends this ...
    signal.signal(signal.SIGTERM, cleanup_pid_file)

    if check_process_file(PID_FILE):
        print(
            "Process ID file already exists. Remove the file if you're sure another instance isn't running with the command: rm bot.pid"
        )
        sys.exit(1)
    else:
        with PID_FILE.open("w") as f:
            f.write(str(os.getpid()))
            print(f"Wrote PID to file {PID_FILE}")
            f.close()
    try:
        if EnvService.get_health_service_enabled():
            try:
                PROCESS = HealthService().get_process()
            except:
                traceback.print_exc()
                print("The health service failed to start.")

        asyncio.get_event_loop().run_until_complete(main())
    except KeyboardInterrupt:
        print("Caught keyboard interrupt, killing and removing PID")
    except Exception as e:
        traceback.print_exc()
        print(str(e))
        print("Removing PID file")
    finally:
        cleanup_pid_file(None, None)

    sys.exit(0)


if __name__ == "__main__":
    sys.exit(init())
