"""
Телеграм бот Элли.
Может сама писать когда хочет.
"""

import asyncio
from telegram import Bot, Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, ELLI_NAME


class TelegramInterface:

    def __init__(self, brain, emotion, memory, drives):
        self.brain = brain
        self.emotion = emotion
        self.memory = memory
        self.drives = drives
        self.bot = Bot(token=TELEGRAM_TOKEN)
        self.app = None
        self.loop = None

    async def send_message(self, text: str):
        """Элли пишет сама"""
        try:
            await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
        except Exception as e:
            print(f"Ошибка отправки: {e}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка сообщения от пользователя"""
        msg = update.message.text

        # Эмоции
        self.emotion.process_input(msg)

        # Память
        memories = self.memory.recall(msg)

        # Потребности
        drives_context = self.drives.get_context()
        self.drives.on_conversation()

        # Контекст
        ctx = self.emotion.get_context() + memories + drives_context

        # Думаем
        answer = self.brain.think(msg, ctx)

        # Обновляем
        self.emotion.process_input(answer)
        self.memory.extract_facts(msg, answer)

        # Отвечаем
        await update.message.reply_text(answer)

    def start_bot(self):
        """Запустить бота в фоне"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.app = Application.builder().token(TELEGRAM_TOKEN).build()
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        # Запускаем в фоновом потоке
        import threading
        thread = threading.Thread(target=self._run_bot, daemon=True)
        thread.start()

    def _run_bot(self):
        """Запуск polling в отдельном потоке"""
        self.loop.run_until_complete(self.app.initialize())
        self.loop.run_until_complete(self.app.start())
        self.loop.run_until_complete(self.app.updater.start_polling())
        self.loop.run_forever()

    def send_proactive_message(self, message: str):
        """Элли пишет первой (из основного потока)"""
        if self.loop:
            future = asyncio.run_coroutine_threadsafe(self.send_message(message), self.loop)
            try:
                future.result(timeout=5)  # Ждём результат
            except Exception as e:
                print(f"Ошибка проактивной отправки: {e}")