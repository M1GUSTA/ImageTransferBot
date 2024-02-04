from aiogram.filters.callback_data import CallbackData
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder


# This is a simple keyboard, that contains 2 buttons
button = [
        [
            InlineKeyboardButton(text="Начать перенос стиля",
                                 callback_data="start_transfer"),
        ],
    ]
keyboard = InlineKeyboardMarkup(
        inline_keyboard=button,
)

