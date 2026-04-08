from aiogram import Router

from handlers_private_callbacks import router as callbacks_router
from handlers_private_messages import router as messages_router

router = Router()
router.include_router(callbacks_router)
router.include_router(messages_router)
