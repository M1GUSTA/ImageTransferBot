from aiogram.fsm.state import StatesGroup, State


class Transfering(StatesGroup):
    pic1 = State()
    pic2 = State()