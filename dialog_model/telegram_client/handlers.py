import json
import os
from collections import defaultdict
from http import HTTPStatus
from typing import Optional

import aiohttp
from aiogram import Dispatcher, types
from aiohttp import ServerDisconnectedError


class HandlersRegister:
    """Class which registers all text generator telegram client handlers."""

    WELCOME_MESSAGE = '*Начало нового диалога*'

    DEFAULT_PARAMS = {
        'n_candidates': 36,
        'max_n_context_tokens': 100,
        'temperature': 1.0,
        'top_k': 0,
        'top_p': 1.0,
        'repetition_penalty': 1.0
    }

    def __init__(
            self,
            dispatcher: Dispatcher,
            dialog_model_url: str,
            dialog_model_auth: Optional
    ):
        self._dispatcher = dispatcher
        self._contexts_cache = ContextsCache()
        self._url = dialog_model_url
        self._auth = dialog_model_auth

    def register_start_message_handler(self):
        """Handles `/start` command and sends welcome message."""

        @self._dispatcher.message_handler(commands=['start'])
        async def start(message: types.Message):
            await message.answer(self.WELCOME_MESSAGE)

    def register_send_reply_message_handler(self):
        """Replies on user input message."""

        @self._dispatcher.message_handler()
        async def send_reply(message: types.Message):
            user_id = _get_user_id_from_message(message)
            self._contexts_cache.add_utterance(message.text, user_id=user_id)
            context = self._contexts_cache.get_context(user_id)
            response_candidates = await self._get_response_candidates(context)
            response = _select_response(response_candidates)
            self._contexts_cache.add_utterance(response, user_id=user_id)
            await message.answer(text=response)

    async def _get_response_candidates(self, context):
        url = os.path.join(self._url, 'response_candidates')

        payload = {'context': context}
        payload.update(self.DEFAULT_PARAMS)

        headers = {'Content-Type': 'application/json'}
        async with aiohttp.ClientSession(auth=self._auth) as session:
            try:
                async with session.post(url=url, data=json.dumps(payload), headers=headers) as response:
                    status = response.status
                    reply = await response.text()
                    return reply, status
            except ServerDisconnectedError:
                return None, HTTPStatus.INTERNAL_SERVER_ERROR

    def register_all_handlers(self):
        self.register_start_message_handler()
        self.register_send_reply_message_handler()


def _select_response(responses):
    responses = sorted(responses, key=lambda response: len(response))
    response = responses[len(responses) // 2]
    return response


def _get_user_id_from_message(message):
    username = str(message.chat['username'])
    chat_id = str(message.chat['id'])
    user_name = "".join(x for x in username if x.isalnum())
    user_id = user_name + '_' + chat_id
    return user_id


class ContextsCache:
    def __init__(self, max_context_len):
        self._cache = defaultdict(list)
        self._max_context_len = max_context_len

    def add_utterance(self, utterance: str, user_id):
        self._cache[user_id].append(utterance)
        self._cache[user_id] = self._cache[user_id][-self._max_context_len:]

    def get_context(self, user_id):
        return self._cache[user_id]
