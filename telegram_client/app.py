import argparse
from collections import defaultdict
import json
import logging

from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
import aiohttp
from log_config import prepare_logging

_logger = logging.getLogger(__name__)


class App:
    _WELCOME_MESSAGE = "*Начало нового диалога*"
    _DIALOG_MODEL_PAYLOAD = {
        "n_candidates": 20,
        "max_n_context_tokens": 70,
        "temperature": 0.9,
        "top_k": 100,
        "top_p": 1,
        "repetition_penalty": 3
    }

    def __init__(self, telegram_api_token, response_candidates_url, max_n_context_messages):
        self._telegram_api_token = telegram_api_token
        self._response_candidates_url = response_candidates_url
        self._contexts_cache = ContextsCache(max_n_context_messages)
        # self._logging_handler = LoggingHandler(logs_dir)

        bot = Bot(token=telegram_api_token)
        self._dispatcher = Dispatcher(bot)

    def register_start_message_handler(self):
        @self._dispatcher.message_handler(commands=['start'])
        async def start(message: types.Message):
            self._contexts_cache.clear_context(_get_user_id_from_message(message))
            await message.answer(self._WELCOME_MESSAGE)

        return start

    def register_reply_on_user_message_handler(self):
        @self._dispatcher.message_handler()
        async def reply_on_user_message(message: types.Message):
            user_id = _get_user_id_from_message(message)

            self._contexts_cache.add_message_text(message_text=message.text, user_id=user_id)

            context = self._contexts_cache.get_context(user_id)
            dialog_model_response = await self._get_dialog_model_response(context)

            self._contexts_cache.add_message_text(message_text=dialog_model_response, user_id=user_id)

            await message.answer(dialog_model_response)

        return reply_on_user_message

    async def _get_dialog_model_response(self, context):
        payload = {'context': context}
        payload.update(self._DIALOG_MODEL_PAYLOAD)
        payload = json.dumps(payload)
        headers = {'Content-Type': 'application/json'}

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url=self._response_candidates_url, data=payload, headers=headers) as response:
                    response = await response.json()
                    response_candidates = response['response_candidates']
                    selected_response_candidate = self._select_response_candidate(response_candidates)
            except aiohttp.ServerDisconnectedError:
                selected_response_candidate = None

        log = {
            'context': context,
            'response_candidates': response_candidates,
            'selected_response_candidate': selected_response_candidate
        }
        log = json.dumps(log, indent=2, ensure_ascii=False)
        _logger.info(log)

        return selected_response_candidate

    @staticmethod
    def _select_response_candidate(response_candidates):
        return response_candidates[0]

    def start(self):
        self.register_start_message_handler()
        self.register_reply_on_user_message_handler()

        executor.start_polling(self._dispatcher, skip_updates=True)


class ContextsCache:
    def __init__(self, max_n_context_messages):
        self._max_n_context_messages = max_n_context_messages
        self._user_id_to_context = defaultdict(list)

    def add_message_text(self, message_text, user_id):
        self._user_id_to_context[user_id].append(message_text)
        self._user_id_to_context[user_id] = self._user_id_to_context[user_id][-self._max_n_context_messages:]

    def get_context(self, user_id):
        return self._user_id_to_context[user_id]

    def clear_context(self, user_id):
        self._user_id_to_context[user_id] = []


def _get_user_id_from_message(message):
    username = str(message.chat['username'])
    chat_id = str(message.chat['id'])
    user_name = "".join(x for x in username if x.isalnum())
    user_id = user_name + '_' + chat_id
    return user_id


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--telegram_api_token', type=str, required=True)
    parser.add_argument('--response_candidates_url', type=str, required=True)
    parser.add_argument('--logs_dir', type=str, required=True)
    parser.add_argument('--max_n_context_messages', type=int, required=False, default=12)

    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    prepare_logging(logs_dir=args.logs_dir)
    app = App(telegram_api_token=args.telegram_api_token,
              response_candidates_url=args.response_candidates_url,
              max_n_context_messages=args.max_n_context_messages)
    app.start()


if __name__ == "__main__":
    main()
