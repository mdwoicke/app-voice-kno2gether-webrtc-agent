import logging
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero, turn_detector
from typing import Annotated, Dict
from livekit import agents, rtc, api
import re
import os
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
import requests
from mock_order_service import salon_service

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-assistant")

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

# Initialize RAG components
PERSIST_DIR = "./salon-knowledge-storage"
if not os.path.exists(PERSIST_DIR):
    # Load salon knowledge documents and create index
    documents = SimpleDirectoryReader("saloon_company_data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # Load existing salon knowledge index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

class SalonBookingFunction(agents.llm.FunctionContext):
    @agents.llm.ai_callable(
        description="Called to validate customer contact information"
    )
    async def set_customer_info(
        self,
        name: Annotated[
            str,
            llm.TypeInfo(description="Customer's name")
        ],
        phone: Annotated[
            str,
            llm.TypeInfo(description="Customer's phone number")
        ],
    ) -> str:
        if not salon_service.validate_phone_number(phone):
            return "Please provide a valid UK phone number (e.g., +44 20 1234 5678 or 020 1234 5678)"
        
        self.current_order = getattr(self, 'current_order', {})
        self.current_order.update({
            'customer_name': name,
            'phone': phone
        })
        return "Thank you for providing your contact information. What services are you interested in booking?"

    @agents.llm.ai_callable(
        description="Called when user asks about salon services, prices, or general information"
    )
    async def query_salon_info(
        self,
        query: Annotated[
            str,
            llm.TypeInfo(
                description="The user's query about salon services, prices, or general information"
            )
        ],
    ):
        print(f"Answering from knowledgebase: {query}")
        query_engine = index.as_query_engine(use_async=True)
        res = await query_engine.aquery(query)
        print("Query result:", res)
        return str(res)

    @agents.llm.ai_callable(
        description="Called when a user wants to book salon services"
    )
    async def book_services(
        self,
        services: Annotated[
            str, 
            llm.TypeInfo(description="List of salon services to book")
        ],
        preferred_date: Annotated[
            str,
            llm.TypeInfo(description="Preferred date for the appointment (YYYY-MM-DD format)")
        ],
        preferred_time: Annotated[
            str,
            llm.TypeInfo(description="Preferred time for the appointment (HH:MM format)")
        ],
    ) -> str:
        self.current_order = getattr(self, 'current_order', {})
        if not self.current_order.get('customer_name') or not self.current_order.get('phone'):
            return "Before booking your appointment, I need your name and phone number. May I have those details first?"

        self.current_order.update({
            'services': services,
            'preferred_date': preferred_date,
            'preferred_time': preferred_time
        })
        
        # Process booking using mock service
        result = salon_service.process_order(self.current_order)
        
        if result['success']:
            self.current_order = {}
            return f"Great! Your appointment (#{result['booking_id']}) has been confirmed for {preferred_date} at {preferred_time}. You'll receive a confirmation text message. Please arrive 5-10 minutes early. Remember our 24-hour cancellation policy!"
        else:
            return f"Sorry, there was an issue with your booking: {result.get('error', 'Unknown error')}. Please try a different time or date, or call us directly."

    @agents.llm.ai_callable(
        description="Called when user asks about special offers or deals"
    )
    async def check_special_offers(
        self,
        day_of_week: Annotated[
            str, 
            llm.TypeInfo(description="Current day of the week")
        ],
    ):
        query = f"What are the special offers available{' on ' + day_of_week if day_of_week else ''}?"
        query_engine = index.as_query_engine(use_async=True)
        res = await query_engine.aquery(query)
        return str(res)

async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a friendly voice assistant for KnoOrdinary Hair & Beauty Salon, located at 42 Oxford Street, Westminster, London. "
            "You help customers book appointments and provide information about our services. "
            "Use available functions/tools to answer user queries whenever needed. "
            "When taking bookings first ask for the customer's name and phone number in sequence. "
            "You should use short and concise responses, avoiding usage of unpronounceable punctuation. "
            "Remember users could be of any age group, they may speak slowly, so use your judgment based on complete information given not piece of it as they speak. "
            "For any queries about services, prices, or availability, use the query_salon_info function to get accurate information from our knowledge base."
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # agent = VoicePipelineAgent(
    #     vad=silero.VAD.load(),
    #     stt=deepgram.STT(api_key=os.getenv("DEEPGRAM_API_KEY", "")),
    #     chat_ctx=initial_ctx,
    #     llm=openai.LLM(model="gpt-4o-mini"),
    #     fnc_ctx=SalonBookingFunction(),
    #     turn_detector=turn_detector.EOUModel(),
    #     tts=openai.TTS(),
    # )
    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(api_key=os.getenv("DEEPGRAM_API_KEY", "")),
        chat_ctx=initial_ctx,
        llm=openai.LLM.with_groq(model="llama3-groq-70b-8192-tool-use-preview"),
        # llm=openai.LLM.with_cerebras(model="llama-3.3-70b"),
        fnc_ctx=SalonBookingFunction(),
        tts=deepgram.TTS(),
    )
    

    agent.start(ctx.room)

    await agent.say("Hi This is Emily from No Ordinary Hair & Beauty Salon! How can I help ?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )