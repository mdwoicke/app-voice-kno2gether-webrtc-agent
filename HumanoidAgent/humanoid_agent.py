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
import re
import os
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
import requests
from mock_order_service import order_service

load_dotenv()
logger = logging.getLogger("voice-assistant")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# Initialize RAG components
PERSIST_DIR = "./pizza-knowledge-storage"
if not os.path.exists(PERSIST_DIR):
    # Load pizza knowledge documents and create index
    documents = SimpleDirectoryReader("pizza_company_data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # Load existing pizza knowledge index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

class PizzaOrderFunction(llm.FunctionContext):
    def __init__(self):
        self.current_order: Dict = {}

    @llm.ai_callable(
        description="Called to validate a UK address and postcode"
    )
    async def validate_address(
        self,
        address: Annotated[
            str,
            llm.TypeInfo(description="Full UK address including postcode")
        ],
    ) -> str:
        # Extract postcode from address
        postcode_match = re.search(r'[A-Z]{1,2}[0-9][A-Z0-9]? ?[0-9][A-Z]{2}', address.upper())
        if not postcode_match:
            return "The address appears to be invalid. Please provide a valid UK address with a proper postcode format (e.g., SW1A 1AA)."
        
        self.current_order['address'] = address
        return "Address validated successfully. What would you like to order?"

    @llm.ai_callable(
        description="Called to validate customer information"
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
        if not order_service.validate_phone_number(phone):
            return "Please provide a valid UK phone number (e.g., +44 20 1234 5678 or 020 1234 5678)"
        
        self.current_order.update({
            'customer_name': name,
            'phone': phone
        })
        return "Customer information recorded. What's your delivery address?"

    @llm.ai_callable(
        description="Called when user asks about menu items, prices, or general information about KnoOrdinary Pizza Center"
    )
    async def query_pizza_info(
        self,
        query: Annotated[
            str,
            llm.TypeInfo(
                description="The user's query about pizza menu, prices, or restaurant information"
            )
        ],
    ):
        print(f"Answering from knowledgebase: {query}")
        query_engine = index.as_query_engine(use_async=True)
        res = await query_engine.aquery(query)
        print("Query result:", res)
        return str(res)

    @llm.ai_callable(
        description="Called when a user wants to place a pizza order"
    )
    async def place_order(
        self,
        pizza_items: Annotated[
            str, 
            llm.TypeInfo(description="List of pizzas and quantities")
        ],
    ) -> str:
        if not self.current_order.get('customer_name') or not self.current_order.get('address') or not self.current_order.get('phone'):
            return "Before placing your order, I need your name, phone number, and delivery address. May I have those details first?"

        self.current_order['items'] = pizza_items
        
        # Process order using mock service
        result = order_service.process_order(self.current_order)
        
        if result['success']:
            # Clear current order after successful placement
            self.current_order = {}
            return f"Great! Your order (#{result['order_id']}) has been confirmed. Your pizzas will be delivered to your address in about 30-45 minutes. You'll receive updates on your phone. Thank you for choosing KnoOrdinary Pizza Center!"
        else:
            return f"Sorry, there was an issue with your order: {result.get('error', 'Unknown error')}. Please try again or call us directly."

    @llm.ai_callable(
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


# This example uses our open-weight turn detection model to detect when the user is
# done speaking. This approach is more accurate than the default VAD model, reducing
# false positive interruptions by the agent.
async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a friendly voice assistant for KnoOrdinary Pizza Center, located at 42 Baker Street, Westminster, London. "
            "You help customers place pizza orders and provide information about our menu and services. "
            "Here's what customers need to know: "
             "- We're open daily from 11:00 AM to 11:00 PM "
            "- We offer both classic and signature pizzas, ranging from £10.99 to £15.99 "
            "- Our most popular pizzas include Margherita (£10.99), Pepperoni Passion (£12.99), and our signature KnoOrdinary Special (£15.99) "
            "- We have great vegetarian options like Veggie Supreme (£11.99) and Mediterranean Delight (£14.99) "
            "- Delivery is free within 3 miles, £2.50 for 3-5 miles, with a minimum order of £15.00 "
            "- We have special offers: Monday Madness (50% off classics), Student Discount (10% off), and Family Deal (2 large pizzas + 2 sides for £35) "
            "When taking orders first ask the customer their name and address in sequence and validate if the address is valid UK Address like if the PostCode is valid format or not. "
            "You should use short and concise responses, avoiding usage of unpronounceable punctuation. "
            "Remember user could be anyone from different age group, they may speak slow, so use your judgement based on complete information given not piece of it as they speak"
        ),
    )


    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
        llm=openai.LLM(model="gpt-4o-mini"),
        turn_detector=turn_detector.EOUModel(),
    )

    agent.start(ctx.room, participant)

    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def _on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)
        usage_collector.collect(mtrcs)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: ${summary}")

    ctx.add_shutdown_callback(log_usage)

    await agent.say("Hi This is Emily from No Ordinary Pizza Center! How can I help ?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )