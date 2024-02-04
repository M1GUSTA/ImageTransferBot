import asyncio
from pathlib import Path

from aiogram import Router, F, Bot
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, CallbackQuery, FSInputFile

from tgbot.keyboards.inline import keyboard
from tgbot.misc.states import Transfering
from tgbot.models.Vgg16Transfer import generate_styled_image

user_router = Router()


@user_router.message(CommandStart())
async def user_start(message: Message):
    await message.answer("Привет, это бот для переноса стиля с одной картинки на другую",
                         reply_markup=keyboard)

@user_router.message(F.text, Command("transfer_style"))
async def user_start(message: Message, state: FSMContext):
    await state.set_state(Transfering.pic1)
    await message.answer("Пришлите фото на котором вы хотите изменить стиль")


@user_router.callback_query(F.data=='start_transfer')
async def start_trans(callback: CallbackQuery, state: FSMContext):
    await state.set_state(Transfering.pic1)
    await callback.message.answer("Пришлите фото на котором вы хотите изменить стиль")

@user_router.message(Transfering.pic1, F.photo)
async def download_pic1(message: Message, state: FSMContext, bot: Bot):
    await bot.download(
        message.photo[-1],
        destination=Path("tgbot", "models", "data", "content1.png")
    )
    a = message.photo[-1].file_id
    await state.set_state(Transfering.pic2)
    await message.answer("Пришлите картинку с которой вы хотите перенести стиль")

@user_router.message(Transfering.pic2, F.photo)
async def download_pic2(message: Message, state: FSMContext, bot: Bot):
    await state.clear()
    await bot.download(
        message.photo[-1],
        destination=Path("tgbot", "models", "data", "style1.png")
    )
    await message.answer("Обработка начата, придётся немного подождать")
    generated_image_task = asyncio.create_task(generate_styled_image())
    await asyncio.gather(generated_image_task)


    res_image = FSInputFile(path=Path('tgbot', 'models','data', "generated_image.png"), filename="generated_image.png")
    result = await message.answer_photo(
        res_image
    )
    # file_id = result.photo[-1].file_id
    # await message.answer("Стилизованная картинка".join(file_id))