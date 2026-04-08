"""
topic_utils.py — единый источник правды для тематических семей, политик генерации,
image-policy и news-трансформации.

Раньше логика была дублирована:
  - content.py: _topic_family(topic, prompt) — своя версия
  - image_search.py: detect_topic_family(query) — своя версия с другими терминами

Теперь все используют эту библиотеку. Расхождений нет.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Canonical topic families with detection terms
# ---------------------------------------------------------------------------

TOPIC_FAMILY_TERMS: dict[str, dict] = {
    "food": {
        "ru": [
            "еда", "блюд", "рецепт", "кухн", "ресторан", "кафе", "кофе", "выпечк", "тест", "деликатес",
            "фуд", "пицц", "суш", "бургер", "десерт", "завтрак", "обед", "ужин", "перекус", "снек",
            "продукт питан", "ингредиент", "соус", "гриль", "барбекю", "веган", "вегетари", "безглютен",
            "фермерск", "органическ", "сыр", "мясо", "рыба", "морепрод", "хлеб", "кондитер",
            "поваренн", "шеф", "повар", "гастроном", "стритфуд", "здоровое питани", "food",
        ],
        "en": [
            "food", "recipe", "cooking", "cuisine", "restaurant", "cafe", "coffee", "bakery",
            "pastry", "gourmet", "pizza", "sushi", "burger", "dessert", "breakfast", "lunch",
            "dinner", "snack", "vegan", "vegetarian", "gluten-free", "organic", "farm-to-table",
            "cheese", "meat", "seafood", "bread", "chef", "gastronomy", "street food", "foodie",
        ],
        "image_queries": [
            "beautifully plated food dish close-up editorial photography",
            "fresh ingredients kitchen flat lay overhead food photography",
            "restaurant dish gourmet plating professional food photo",
            "homemade cooking baking kitchen natural light editorial",
            "coffee cup cafe morning mood editorial photography",
        ],
        "allowed_visual_classes": ["food", "dish", "kitchen", "ingredients", "restaurant", "cafe", "cooking", "beverage", "product"],
        "blocked_visual_classes": ["tech", "code", "server", "abstract circuit", "corporate meeting", "finance chart"],
        "news_angle": (
            "Раскрой эту новость через призму потребителя и гастрономической культуры. "
            "Какой продукт или тренд меняется? Что это значит для любителей еды, ресторанов, производителей? "
            "Используй чувственный, практичный язык. Не пиши как технический обзор."
        ),
    },
    "health": {
        "ru": [
            "здоровь", "медицин", "врач", "лечен", "диагноз", "симптом", "болезн",
            "здоровый образ жизни", "зож", "диет", "витамин", "спорт", "фитнес",
            "тренировк", "психолог", "ментальн", "сон", "стресс", "иммунитет", "профилактик",
            "реабилитац", "физиотерапи", "нутрицион", "биохак", "longevity", "превентивн",
            "аптек", "бад", "supplement", "wellness", "медит", "дыхание", "йог", "питани",
        ],
        "en": [
            "health", "medicine", "doctor", "treatment", "diagnosis", "symptom",
            "healthy lifestyle", "nutrition", "diet", "vitamin", "fitness", "workout", "psychology",
            "mental health", "sleep", "stress", "immunity", "prevention", "rehabilitation",
            "physiotherapy", "nutritionist", "biohacking", "longevity", "preventive",
            "pharmacy", "supplement", "wellness", "meditation", "breathing", "yoga",
        ],
        "image_queries": [
            "wellness healthy lifestyle natural light editorial photo",
            "fitness workout exercise motivation editorial photography",
            "healthy food nutrition meal prep clean editorial",
            "meditation mindfulness peaceful nature editorial",
            "medical professional consultation warm editorial photo",
        ],
        "allowed_visual_classes": ["wellness", "fitness", "nature", "medical", "calm", "exercise", "herbs"],
        "blocked_visual_classes": ["surgery graphic", "illness graphic", "pharmaceutical ad", "tech abstract", "corporate", "food", "restaurant", "cuisine", "dish"],
        "news_angle": (
            "Раскрой эту новость через практические последствия для здоровья людей. "
            "Какие реальные изменения это означает для пациентов, специалистов или тех, кто следит за здоровьем? "
            "Избегай медицинских обещаний и сенсационности. Придерживайся доказательного, осторожного тона."
        ),
    },
    "beauty": {
        "ru": [
            "красот", "косметик", "косметолог", "уход", "макияж", "маникюр", "педикюр", "ресниц", "бров",
            "парикмахер", "стриж", "окраш", "укладк", "spa", "спа", "антивозраст", "омолож",
            "кремы", "сыворотк", "мицеллярн", "тонер", "маск", "пилинг", "скраб",
            "дерматолог", "инъекц", "ботокс", "филлер", "мезотерапи", "клиника красот",
            "нейл", "гель лак", "шеллак", "перманентн", "татуаж", "бьюти", "салон красот",
        ],
        "en": [
            "beauty", "cosmetics", "skincare", "makeup", "manicure", "pedicure", "lashes",
            "brows", "hairdresser", "haircut", "coloring", "blowout", "spa", "anti-aging",
            "rejuvenation", "cream", "serum", "micellar", "toner", "mask", "peeling", "scrub",
            "cosmetologist", "dermatologist", "botox", "filler", "mesotherapy",
            "nail", "gel polish", "shellac", "permanent makeup", "beauty salon",
        ],
        "image_queries": [
            "beauty skincare product flat lay editorial photography",
            "cosmetics makeup artistic editorial close-up photo",
            "spa salon beauty treatment process editorial",
            "hair styling beauty salon professional editorial",
            "nail art manicure creative beauty editorial",
        ],
        "allowed_visual_classes": ["beauty product", "skincare", "makeup", "salon", "spa", "cosmetics", "hair", "nail"],
        "blocked_visual_classes": ["tech abstract", "corporate", "server", "random glamour stock photo", "finance chart"],
        "news_angle": (
            "Раскрой эту новость через призму потребителя красоты и бьюти-индустрии. "
            "Что меняется в уходе, процедурах или продуктах? Какова практическая ценность для клиентов салонов или покупателей косметики? "
            "Используй живой, конкретный язык без корпоративного пафоса."
        ),
    },
    "local_business": {
        "ru": [
            "ремонт", "сервис", "мастер", "мастерск", "услуг", "локальн", "местн", "город",
            "ателье", "химчист", "автосервис", "шиномонтаж", "электрик", "сантехник", "строительств",
            "отделк", "дизайн интерьер", "мебель на заказ", "клининг", "уборк",
            "охран", "монтаж", "установк", "ремонт техник", "ноутбук ремонт", "смартфон ремонт",
            "доставк", "курьер", "малый бизнес",
        ],
        "en": [
            "repair", "service", "local", "city", "master", "workshop", "tailor", "dry cleaning",
            "auto service", "tire", "electrician", "plumber", "construction", "renovation",
            "interior design", "custom furniture", "cleaning", "security", "installation",
            "laptop repair", "phone repair", "delivery", "courier", "small business",
        ],
        "image_queries": [
            "local business workshop tools craftsmanship editorial photo",
            "repair service technician workspace professional editorial",
            "small business owner shop storefront editorial photography",
            "service process workplace hands-on professional editorial",
            "local craftsman skilled work close-up editorial photo",
        ],
        "allowed_visual_classes": ["tools", "workshop", "workplace", "service", "craftsmanship", "store", "hands-on"],
        "blocked_visual_classes": ["corporate abstract", "tech circuit", "finance chart", "random glamour"],
        "news_angle": (
            "Раскрой эту новость через призму малого и местного бизнеса. "
            "Что это изменит для предпринимателей, клиентов, локального рынка? "
            "Конкретные практические последствия важнее абстрактных рассуждений."
        ),
    },
    "education": {
        "ru": [
            "образован", "учёб", "школ", "университет", "курс", "обучен", "урок", "учитель",
            "преподаватель", "студент", "ученик", "репетитор", "егэ", "огэ", "олимпиад",
            "дистанционн", "онлайн-курс", "самообразован", "навык", "компетенц", "диплом",
            "профессионал развитие", "повышен квалификац", "тренинг", "вебинар", "лекц",
            "менторств", "коучинг", "edtech", "педагог",
        ],
        "en": [
            "education", "school", "university", "course", "learning", "lesson", "teacher",
            "professor", "student", "tutor", "exam", "olympiad", "distance learning",
            "online course", "self-education", "skill", "competence", "diploma",
            "professional development", "training", "webinar", "lecture",
            "mentorship", "coaching", "edtech", "pedagogy",
        ],
        "image_queries": [
            "education learning study books desk editorial photography",
            "online learning student laptop course editorial photo",
            "teacher classroom education engagement editorial",
            "library books knowledge learning editorial photograph",
            "skill development training workshop editorial photo",
        ],
        "allowed_visual_classes": ["study", "books", "learning", "classroom", "knowledge", "skill", "notebook"],
        "blocked_visual_classes": ["corporate abstract", "server rack", "finance chart", "random glamour"],
        "news_angle": (
            "Раскрой эту новость через призму обучения и развития. "
            "Как это влияет на образовательный процесс, учеников или преподавателей? "
            "Какой практический навык или знание из этого можно извлечь? Ясность и применимость — на первом месте."
        ),
    },
    "finance": {
        "ru": [
            "финанс", "деньги", "инвестиц", "акции", "облигац", "дивиденд", "доходност",
            "портфель", "брокер", "биржа", "крипто", "биткоин", "эфириум", "нфт", "nft",
            "банк", "кредит", "ипотек", "вклад", "страховк", "налог", "бухгалтер",
            "пенсион", "финансов планирован", "бюджет", "экономи", "экономика",
            "валют", "курс доллар", "курс евро", "forex", "трейдинг", "фондов рынок",
        ],
        "en": [
            "finance", "money", "investment", "stocks", "bonds", "dividend", "yield",
            "portfolio", "broker", "exchange", "crypto", "bitcoin", "ethereum", "nft",
            "bank", "credit", "mortgage", "deposit", "insurance", "tax", "accounting",
            "pension", "financial planning", "budget", "economy", "economics",
            "currency", "forex", "trading", "stock market",
        ],
        "image_queries": [
            "financial planning budget chart business editorial photography",
            "investment portfolio finance dashboard professional editorial",
            "money economy business context editorial photo",
            "financial charts graphs planning context editorial",
            "banking finance professional context editorial photograph",
        ],
        "allowed_visual_classes": ["finance", "chart", "planning", "business context", "banking", "economy"],
        "blocked_visual_classes": ["food", "beauty", "fitness", "random glamour", "tech abstract circuit"],
        "news_angle": (
            "Раскрой эту новость через призму влияния на финансы, рынки или личные деньги людей. "
            "Какой практический вывод для инвестора, потребителя или бизнеса? "
            "Избегай безответственных обещаний и спекулятивных заявлений."
        ),
    },
    "marketing": {
        "ru": [
            "маркетинг", "smm", "контент", "реклам", "таргет", "бренд", "pr", "пиар",
            "продвижен", "трафик", "конверс", "лидогенерац", "воронк", "email-маркетинг",
            "seo", "контекстн", "инфлюенсер", "блогер", "коллаборац", "амбассадор",
            "копирайтинг", "тексты для бизнес", "продающ", "стратег", "позициониров",
            "аналитик аудитор", "telegram-маркетинг", "телеграм канал", "монетизац",
        ],
        "en": [
            "marketing", "smm", "content", "advertising", "targeting", "brand", "pr", "promotion",
            "traffic", "conversion", "lead generation", "funnel", "email marketing",
            "seo", "ppc", "influencer", "blogger", "collaboration", "ambassador",
            "copywriting", "sales copy", "strategy", "positioning", "audience analytics",
            "telegram marketing", "monetization",
        ],
        "image_queries": [
            "marketing strategy analytics dashboard professional editorial",
            "content creation social media workspace editorial photo",
            "brand identity design creative process editorial",
            "digital marketing campaign workspace professional editorial",
            "analytics metrics dashboard business planning editorial",
        ],
        "allowed_visual_classes": ["analytics", "dashboard", "creative workspace", "content creation", "brand", "strategy"],
        "blocked_visual_classes": ["food", "medical", "fitness unrelated", "random nature"],
        "news_angle": (
            "Раскрой эту новость через призму маркетинга и продвижения. "
            "Как это влияет на рекламодателей, SMM-специалистов или бренды? "
            "Практические последствия и конкретные инструменты важнее абстрактного анализа."
        ),
    },
    "lifestyle": {
        "ru": [
            "лайфстайл", "образ жизни", "личн развитие", "самопознан", "привычк",
            "продуктивност", "тайм-менеджмент", "мотивац", "вдохновен", "mindfulness",
            "путешестви", "отдых", "хобби", "творчеств", "рукодел", "дизайн", "интерьер",
            "мода", "стиль", "одежд", "аксессуар", "дом", "уют", "семь", "отношен",
            "dating", "знакомств", "дети", "воспитан", "парентинг",
        ],
        "en": [
            "lifestyle", "personal development", "self-improvement", "habits",
            "productivity", "time management", "motivation", "inspiration", "mindfulness",
            "travel", "vacation", "hobby", "creativity", "crafts", "design", "interior",
            "fashion", "style", "clothing", "accessories", "home", "comfort", "family",
            "relationships", "dating", "children", "parenting",
        ],
        "image_queries": [
            "lifestyle morning routine cozy home editorial photography",
            "personal development motivation book coffee editorial photo",
            "travel adventure lifestyle editorial photography",
            "creative hobby workspace artistic editorial photo",
            "family lifestyle warm home editorial photograph",
        ],
        "allowed_visual_classes": ["lifestyle", "home", "cozy", "travel", "creative", "fashion", "nature", "personal"],
        "blocked_visual_classes": ["corporate abstract", "server rack", "medical graphic", "finance chart"],
        "news_angle": (
            "Раскрой эту новость через призму личной жизни, привычек и благополучия. "
            "Как это влияет на повседневный образ жизни? "
            "Живой, человеческий язык — без корпоративного тона и технического жаргона."
        ),
    },
    "expert_blog": {
        "ru": [
            "эксперт", "специалист", "профессионал", "блог эксперт", "авторск", "личн бренд",
            "консультант", "аналитик", "исследовател", "автор", "мнение эксперт",
            "разбор", "обзор", "колонк", "инсайт", "экспертиз",
        ],
        "en": [
            "expert", "specialist", "professional", "expert blog", "author", "personal brand",
            "consultant", "analyst", "researcher", "opinion", "insight", "expertise",
            "commentary", "column", "deep dive", "breakdown",
        ],
        "image_queries": [
            "professional expert workspace editorial photography",
            "thoughtful person writing working editorial photo",
            "book knowledge expert professional editorial",
            "consulting professional meeting editorial photography",
            "author writing workspace creative editorial photo",
        ],
        "allowed_visual_classes": ["professional", "workspace", "books", "editorial", "knowledge", "thinking"],
        "blocked_visual_classes": ["random glamour", "food unrelated", "tech abstract circuit"],
        "news_angle": (
            "Раскрой эту новость через экспертную призму автора канала. "
            "Какой профессиональный вывод можно сделать? Как это влияет на область экспертизы? "
            "Авторская позиция и глубина важнее пересказа заголовков."
        ),
    },
    "massage": {
        "ru": ["массаж", "самомассаж", "массажист", "шея", "спина", "осанк", "плеч", "реабил", "восстанов"],
        "en": ["massage", "massage therapist", "bodywork", "therapist", "masseur"],
        "image_queries": [
            "therapeutic massage hands back treatment editorial photo",
            "massage therapist hands shoulders back recovery photo",
            "body massage calm interior professional therapist realistic photo",
            "massage session treatment closeup hands body realistic photo",
        ],
        "allowed_visual_classes": ["massage", "wellness", "therapy", "hands", "spa", "recovery"],
        "blocked_visual_classes": ["tech", "corporate", "finance chart", "food", "fashion"],
        "news_angle": (
            "Раскрой эту новость через призму восстановления и физического благополучия. "
            "Как это влияет на практику массажа, клиентов или специалистов по восстановлению?"
        ),
    },
    "cars": {
        "ru": ["машин", "авто", "автомоб", "электромоб", "тесла", "двигател"],
        "en": ["car", "cars", "automotive", "vehicle", "dashboard", "interior"],
        "image_queries": [
            "cars automotive exterior editorial photo",
            "cars automotive dashboard interior editorial photo",
            "electric cars automotive industry editorial",
            "modern car interior steering wheel dashboard",
        ],
        "allowed_visual_classes": ["car", "automotive", "vehicle", "road", "transport"],
        "blocked_visual_classes": ["fashion", "food", "tech circuit", "medical"],
        "news_angle": (
            "Раскрой эту новость через призму автомобильного рынка и водителей. "
            "Что меняется для покупателей, автовладельцев или автоиндустрии?"
        ),
    },
    "gaming": {
        "ru": ["игр", "гейм", "игрок", "консоль", "геймпад", "киберспорт", "steam", "playstation", "xbox"],
        "en": ["gaming", "game", "games", "controller", "console", "esports"],
        "image_queries": [
            "gaming setup controller monitor editorial photo",
            "video games gaming desk setup editorial",
            "gaming technology setup controller screen",
            "console gaming controller led setup photo",
        ],
        "allowed_visual_classes": ["gaming", "controller", "console", "esports", "monitor", "setup"],
        "blocked_visual_classes": ["food", "beauty", "medical", "random nature"],
        "news_angle": (
            "Раскрой эту новость через призму игрового сообщества и геймеров. "
            "Что меняется в игровом опыте, экосистеме или индустрии?"
        ),
    },
    "hardware": {
        "ru": ["компьют", "ноут", "ноутбук", "пк", "процессор", "видеокарт", "ssd", "памят", "желез"],
        "en": ["computer", "laptop", "hardware", "pc", "workstation", "desk"],
        "image_queries": [
            "computer hardware pc setup workstation technology desk",
            "laptop workspace technology desk editorial photo",
            "computer components motherboard graphics card editorial",
            "home office computer setup modern desk",
        ],
        "allowed_visual_classes": ["computer", "hardware", "desk", "workspace", "tech device"],
        "blocked_visual_classes": ["food", "beauty", "medical", "fashion", "random glamour"],
        "news_angle": (
            "Раскрой эту новость через призму пользователей ПК и железа. "
            "Что реально меняется в производительности, цене или выборе комплектующих?"
        ),
    },
    "tech": {
        "ru": [
            "технолог", "программ", "разработ", "софт", "код", "ии", "нейросет",
            "искусствен", "ai", "ml", "devops", "облак", "сервер", "баз данн",
            "api", "микросервис", "архитектур", "бэкенд", "фронтенд", "деплой",
            "kubernetes", "docker", "cicd", "saas", "стартап", "data science",
            "python", "javascript", "typescript", "react", "angular", "vue",
            "кибербез", "информац", "блокчейн", "крипт", "финтех", "no-code",
            "low-code", "автоматиз", "робот", "чат-бот", "gpt", "llm",
        ],
        "en": [
            "technology", "software", "programming", "developer", "coding",
            "ai", "machine learning", "devops", "cloud", "server", "database",
            "api", "microservice", "backend", "frontend", "deploy",
            "kubernetes", "docker", "saas", "startup", "data science",
            "python", "javascript", "typescript", "react", "angular",
            "cybersecurity", "blockchain", "crypto", "fintech", "automation",
            "chatbot", "gpt", "llm", "neural", "deep learning",
        ],
        "image_queries": [
            "technology software development workspace monitor code",
            "programming developer laptop code screen editorial",
            "cloud computing server infrastructure technology",
            "artificial intelligence technology abstract data",
            "software development team office workspace",
        ],
        "allowed_visual_classes": ["tech", "code", "server", "developer workspace", "cloud", "circuit"],
        "blocked_visual_classes": ["food", "beauty", "fashion glamour", "medical graphic"],
        "news_angle": (
            "Раскрой эту новость через призму технологий и разработки. "
            "Что меняется в продукте, инфраструктуре или экосистеме?"
        ),
    },
    "business": {
        "ru": [
            "бизнес", "предприниматель", "компан", "маркетинг", "продаж",
            "менеджмент", "управлен", "стратег", "инвестиц", "финанс",
            "аналитик", "метрик", "kpi", "roi", "конверс", "трафик",
            "smm", "контент-маркетинг", "бренд", "реклам", "таргет",
            "воронк", "лидогенерац", "crm", "b2b", "b2c", "ecommerce",
        ],
        "en": [
            "business", "entrepreneur", "company", "marketing", "sales",
            "management", "strategy", "investment", "finance", "analytics",
            "metrics", "kpi", "roi", "conversion", "traffic",
            "smm", "content marketing", "brand", "advertising", "targeting",
            "funnel", "lead generation", "crm", "ecommerce",
        ],
        "image_queries": [
            "business strategy meeting office professional",
            "marketing analytics dashboard data screen",
            "business office workspace professional environment",
            "corporate team meeting presentation boardroom",
        ],
        "allowed_visual_classes": ["business", "office", "analytics", "professional", "strategy"],
        "blocked_visual_classes": ["food", "beauty", "fitness", "random glamour"],
        "news_angle": (
            "Раскрой эту новость через призму бизнеса и предпринимательства. "
            "Как это влияет на рынок, компании или бизнес-процессы?"
        ),
    },
}

# ---------------------------------------------------------------------------
# Family-level tone and guardrail rules for text generation
# ---------------------------------------------------------------------------

FAMILY_GENERATION_RULES: dict[str, dict] = {
    "food": {
        "tone": "sensory, practical, concrete, warm, appetizing",
        "guardrails": (
            "Используй чувственный, конкретный, практичный язык — запахи, вкусы, текстуры. "
            "Запрещён технический жаргон, корпоративный язык, абстрактные рассуждения. "
            "Не пиши про ИИ, технологии или бизнес-метрики — если это не прямая тема поста. "
            "Фокус: конкретное блюдо, рецепт, продукт, ресторан, опыт гостя."
        ),
        "banned_patterns": [
            "в эпоху технологий", "цифровая трансформация", "ai-решение",
            "оптимизация процессов", "синергия", "бизнес-процесс",
        ],
        "cta_style": "попробуй, приготовь, зайди, забронируй столик, расскажи что приготовишь",
        "evidence_level": "sensory and practical — конкретные ингредиенты, технологии, вкусы",
        "post_angles": [
            "рецепт с историей ингредиента",
            "секрет шефа / лайфхак на кухне",
            "почему это блюдо популярно именно сейчас",
            "история происхождения блюда",
            "разбор продукта — как выбрать и не ошибиться",
            "сезонный тренд в меню",
        ],
    },
    "health": {
        "tone": "calm, evidence-based, practical, supportive, cautious",
        "guardrails": (
            "Пиши спокойно, доказательно, без сенсационности и медицинских обещаний. "
            "Не делай заявлений о лечении, гарантиях результата или чудодейственных эффектах. "
            "Не пугай читателя. Приводи практические советы с оговорками при необходимости. "
            "Уважительный тон к читателю, который заботится о своём здоровье."
        ),
        "banned_patterns": [
            "гарантированное лечение", "полностью излечит", "чудодейственный эффект",
            "врачи скрывают", "официальная медицина против", "100% результат",
        ],
        "cta_style": "проконсультируйся, попробуй, обрати внимание, начни с малого, поделись опытом",
        "evidence_level": "evidence-based — ссылки на исследования или практику, с оговорками",
        "post_angles": [
            "практический совет для повседневного здоровья",
            "разбор распространённого мифа",
            "что говорит наука / исследования",
            "признаки, на которые стоит обратить внимание",
            "привычка, которую легко внедрить",
            "взгляд специалиста на актуальный вопрос",
        ],
    },
    "beauty": {
        "tone": "friendly, inspiring, practical, aesthetic, personal",
        "guardrails": (
            "Пиши как близкий бьюти-эксперт, а не как корпоративный пресс-релиз. "
            "Запрещён технический жаргон, абстрактный корпоративный язык, случайные финансовые метрики. "
            "Конкретные продукты, процедуры, результаты, ощущения — вот что важно. "
            "Не обещай невозможных результатов — честность строит доверие."
        ),
        "banned_patterns": [
            "оптимизация бизнес-процессов", "roi маркетинга", "цифровая трансформация",
            "в эпоху ИИ", "алгоритмы красоты", "искусственный интеллект",
        ],
        "cta_style": "попробуй, запишись, расскажи о своём опыте, сохрани себе, покажи результат",
        "evidence_level": "personal + practical — конкретный результат, отзывы, процесс",
        "post_angles": [
            "обзор процедуры / продукта с личным опытом",
            "разбор состава или технологии — понятным языком",
            "до/после без фотошопа — честный взгляд",
            "сезонный уход: что менять и почему",
            "тренд в бьюти — есть ли смысл следовать",
            "советы специалиста по уходу в домашних условиях",
        ],
    },
    "local_business": {
        "tone": "trustworthy, practical, process-oriented, local, human",
        "guardrails": (
            "Акцент на доверии, конкретном процессе и практической ценности для местного клиента. "
            "Никаких абстрактных корпоративных рассуждений — только живой опыт мастера и реальные сценарии. "
            "Покажи процесс, компетентность и заботу о клиенте. "
            "Простой, понятный язык без технического жаргона."
        ),
        "banned_patterns": [
            "глобальная экосистема", "цифровая трансформация", "roi", "конверсионная воронка",
            "b2b решения", "синергия", "нейросеть для бизнеса",
        ],
        "cta_style": "позвони, запишись, приходи, оставь заявку, покажи соседям",
        "evidence_level": "practical + social proof — кейсы, отзывы клиентов, портфолио",
        "post_angles": [
            "кейс — реальная задача клиента и как решили",
            "как мы работаем — прозрачность процесса",
            "частая ошибка клиентов и как её избежать",
            "почему стоит выбрать специалиста, а не делать самому",
            "сезонный совет для клиентов",
            "история из практики — живой опыт мастера",
        ],
    },
    "education": {
        "tone": "clear, structured, useful, empowering, intelligent",
        "guardrails": (
            "Ясность, структура и практическая применимость — основа. "
            "Не перегружай терминологией. Объясняй сложное просто. "
            "Не поучай свысока — помогай и направляй. "
            "Конкретный навык или знание важнее абстрактных рассуждений."
        ),
        "banned_patterns": [
            "это очевидно всем", "любой знает что", "как уже все понимают",
            "ещё раз повторю для непонятливых",
        ],
        "cta_style": "попробуй, примени, проверь себя, поделись с учениками, начни с этого шага",
        "evidence_level": "structured + practical — примеры, упражнения, шаги",
        "post_angles": [
            "конкретный навык — как освоить за N шагов",
            "разбор распространённой ошибки учеников",
            "нестандартный способ объяснить сложное",
            "история успеха ученика / студента",
            "подборка ресурсов по теме",
            "ответ на частый вопрос из практики",
        ],
    },
    "finance": {
        "tone": "honest, analytical, practical, cautious, clear",
        "guardrails": (
            "Пиши честно и аналитически. Избегай безответственных финансовых обещаний. "
            "Не рекламируй конкретные активы как «беспроигрышные». "
            "Практическая польза важнее сенсационности. "
            "Если говоришь о рисках — называй их прямо."
        ),
        "banned_patterns": [
            "100% прибыль гарантирована", "секретная стратегия богатых",
            "вложи и получай пассивный доход без риска", "инсайдерская информация",
        ],
        "cta_style": "разберись, посчитай, проверь свой портфель, проконсультируйся с финансистом",
        "evidence_level": "analytical + data-driven — цифры, источники, сравнения",
        "post_angles": [
            "разбор финансового инструмента — плюсы и минусы",
            "как работает механизм — понятно для неспециалиста",
            "типичная ошибка в личных финансах",
            "что изменилось на рынке и что это значит",
            "практический шаг к финансовой грамотности",
            "взгляд аналитика на актуальный тренд",
        ],
    },
    "marketing": {
        "tone": "practical, strategic, data-informed, creative, direct",
        "guardrails": (
            "Конкретные инструменты, метрики и стратегии — без пустого теоретизирования. "
            "Используй реальные примеры. Не обещай магических результатов без усилий. "
            "Честный взгляд профессионала важнее вдохновляющего, но пустого контента."
        ),
        "banned_patterns": [
            "секретный способ стать вирусным", "этот трюк изменит ваш маркетинг навсегда",
            "простой способ привлечь миллион подписчиков",
        ],
        "cta_style": "применяй, тестируй, измеряй, поделись своим кейсом",
        "evidence_level": "practical + case-based — реальные цифры, кейсы, A/B тесты",
        "post_angles": [
            "разбор рекламной кампании — что сработало и почему",
            "инструмент / тактика — как использовать прямо сейчас",
            "ошибка, которая убивает конверсию",
            "тренд в SMM / контенте — стоит ли следовать",
            "кейс из практики — с цифрами",
            "мнение о новом алгоритме платформы",
        ],
    },
    "lifestyle": {
        "tone": "warm, personal, inspiring, human, relatable",
        "guardrails": (
            "Человеческий, живой, личный тон. Реальные истории и опыт важнее абстрактных советов. "
            "Не поучай и не навязывай единственно верный образ жизни. "
            "Вдохновляй без давления. Конкретные детали делают текст живым."
        ),
        "banned_patterns": [
            "обязан", "должен", "единственный правильный путь", "все успешные люди делают так",
            "если ты этого не делаешь то ты неудачник",
        ],
        "cta_style": "попробуй, расскажи свою историю, поделись, вдохнови кого-то рядом",
        "evidence_level": "personal + relatable — личный опыт, истории, наблюдения",
        "post_angles": [
            "личная история и что она изменила",
            "привычка, которую стоит попробовать",
            "наблюдение о повседневной жизни",
            "рекомендация книги / фильма / места",
            "честный взгляд на популярный тренд",
            "рефлексия о важном",
        ],
    },
    "expert_blog": {
        "tone": "authoritative, thoughtful, original, deep, professional",
        "guardrails": (
            "Авторский голос и экспертная глубина — главное. "
            "Уникальная точка зрения важнее пересказа общеизвестного. "
            "Ссылайся на реальный опыт и практику. "
            "Не копируй стиль новостного агрегатора — это личный авторский канал."
        ),
        "banned_patterns": [
            "как все уже знают", "общеизвестный факт что",
            "все эксперты говорят",
        ],
        "cta_style": "поделись мнением, задай вопрос, расскажи свой опыт",
        "evidence_level": "expertise + original insight — личный опыт, аналитика, примеры из практики",
        "post_angles": [
            "авторский взгляд на актуальную тему в нише",
            "разбор распространённого заблуждения",
            "опыт из практики — что на самом деле работает",
            "нестандартный вывод из привычной ситуации",
            "честная позиция по спорному вопросу",
            "мини-кейс или история из профессиональной жизни",
        ],
    },
    "massage": {
        "tone": "calm, practical, professional, trust-building",
        "guardrails": (
            "Пиши как сильный практик для реальной аудитории массажиста. "
            "Запрещены футуризм, нелепые технологии, странные хайповые примеры, TikTok-мемы. "
            "Нужны земные темы: напряжение, восстановление, боль в шее и спине, сидячая работа, "
            "тренировки, усталость, привычки после сеанса, доверие к специалисту, понятный результат."
        ),
        "banned_patterns": ["ИИ массаж", "роботы заменят массажиста", "революция в массаже"],
        "cta_style": "запишись, попробуй, почувствуй разницу, расскажи близким",
        "evidence_level": "practical + physical — ощущения, результаты, реальные сценарии",
        "post_angles": [
            "узнаваемый бытовой сценарий — когда массаж нужен",
            "объяснение пользы без пафоса",
            "лайфхак для самовосстановления между сеансами",
            "разбор: кому подходит эта техника",
            "история клиента без излишней сентиментальности",
        ],
    },
    "cars": {
        "tone": "practical, informed, no-hype, driver-focused",
        "guardrails": (
            "Не выдумывай футуристические сенсации и не пиши как жёлтая новость. "
            "Нужен практичный взгляд: выбор, удобство, расходы, полезные изменения, реальные сценарии использования."
        ),
        "banned_patterns": ["революция в автопроме", "конец бензиновой эры навсегда"],
        "cta_style": "сравни, проверь, тест-драйв, расскажи свой опыт",
        "evidence_level": "practical + specs — реальные характеристики, цены, сравнения",
        "post_angles": [
            "выбор автомобиля — что реально важно",
            "разбор модели: плюсы и минусы",
            "лайфхак по обслуживанию и экономии",
            "тренд в автопроме — стоит ли реагировать",
            "сравнение: бензин vs электро в реальных условиях",
        ],
    },
    "gaming": {
        "tone": "energetic, community-focused, honest, gamer-voice",
        "guardrails": (
            "Не пиши пустой хайп и не делай вид, что всё — революция. "
            "Делай пост полезным: что меняется для игрока, где выгода, где риск, что реально интересно."
        ),
        "banned_patterns": ["это изменит гейминг навсегда", "революция игровой индустрии"],
        "cta_style": "поиграй, поделись мнением, оцени, расскажи гильдии",
        "evidence_level": "gameplay + community — отзывы игроков, геймплей, опыт",
        "post_angles": [
            "обзор новинки — честно и по делу",
            "сравнение: стоит ли апгрейд",
            "лайфхак в игре / настройке системы",
            "что реально изменилось в обновлении",
            "выбор игры по настроению",
        ],
    },
    "hardware": {
        "tone": "informed, analytical, practical, no-hype",
        "guardrails": (
            "Избегай маркетингового пафоса и пустого пересказа характеристик. "
            "Нужны понятные выводы: что выбрать, за что переплачивают, где реальная польза."
        ),
        "banned_patterns": ["революция в железе", "этот чип изменит всё навсегда"],
        "cta_style": "сравни, посмотри бенчмарк, выбери, расскажи что взял",
        "evidence_level": "specs + benchmarks — реальные тесты, сравнения, соотношение цена/качество",
        "post_angles": [
            "стоит ли брать: честный разбор",
            "что реально важно в характеристиках",
            "апгрейд vs покупка нового",
            "лучшее соотношение цены и производительности прямо сейчас",
            "разбор компонента для конкретной задачи",
        ],
    },
    "tech": {
        "tone": "informed, analytical, developer-friendly, balanced",
        "guardrails": (
            "Пиши для технически грамотной аудитории. Конкретные технические детали допустимы. "
            "Не упрощай чрезмерно, но и не перегружай жаргоном без нужды. "
            "Release/update/infrastructure/product angles приветствуются."
        ),
        "banned_patterns": ["технологии будущего изменят всё", "ИИ заменит всех людей"],
        "cta_style": "попробуй, задеплой, форкни, поделись в команде",
        "evidence_level": "technical + practical — код, архитектура, performance, use cases",
        "post_angles": [
            "разбор новой технологии / библиотеки",
            "практический кейс применения инструмента",
            "сравнение подходов с реальными trade-offs",
            "обновление / релиз — что реально изменилось",
            "архитектурное решение — плюсы и минусы",
        ],
    },
    "business": {
        "tone": "strategic, practical, professional, results-oriented",
        "guardrails": (
            "Конкретные стратегии и результаты важнее мотивационных лозунгов. "
            "Реальные цифры и кейсы строят доверие. "
            "Не обещай быстрых результатов без усилий."
        ),
        "banned_patterns": ["секрет успешного бизнеса который скрывают", "пассивный доход без вложений"],
        "cta_style": "примени, посчитай roi, попробуй в своём бизнесе, поделись результатом",
        "evidence_level": "data + case studies — метрики, кейсы, ROI",
        "post_angles": [
            "бизнес-кейс — что сработало и почему",
            "разбор ошибки — как не наступить на грабли",
            "инструмент / подход — как применить прямо сейчас",
            "тренд рынка — что это значит для вашего бизнеса",
            "практический совет по управлению или маркетингу",
        ],
    },
}

# Default rules for unknown/generic families
FAMILY_GENERATION_RULES["generic"] = {
    "tone": "concrete, human, practical",
    "guardrails": (
        "Пиши приземлённо, конкретно и по-человечески. "
        "Никаких нелепых сенсаций, искусственного хайпа и выдуманных будущих сценариев."
    ),
    "banned_patterns": ["это изменит всё", "революция навсегда"],
    "cta_style": "попробуй, расскажи, поделись мнением",
    "evidence_level": "practical — конкретные примеры и наблюдения",
    "post_angles": [
        "практический совет по теме",
        "история или кейс из жизни",
        "разбор распространённого заблуждения",
        "наблюдение по актуальной теме",
    ],
}

# ---------------------------------------------------------------------------
# Post visual-type classification
# ---------------------------------------------------------------------------

VISUAL_TYPE_PRODUCT = "product"
VISUAL_TYPE_INFRASTRUCTURE = "infra"
VISUAL_TYPE_ANALYTICS = "analytics"
VISUAL_TYPE_TUTORIAL = "tutorial"
VISUAL_TYPE_NEWS = "news"
VISUAL_TYPE_EDITORIAL = "editorial"
VISUAL_TYPE_CASE_STUDY = "case_study"
VISUAL_TYPE_TEXT_ONLY = "text_only"


def _normalize(text: str) -> str:
    """Нормализует текст для матчинга: lowercase, убирает ё, схлопывает пробелы."""
    text = (text or "").strip().lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text)


def detect_topic_family(text: str) -> str:
    """
    Определяет тематическую семью по тексту.

    Возвращает одну из ключей TOPIC_FAMILY_TERMS или "generic".
    Порядок проверки важен — более специфичные семьи стоят первыми.
    """
    q = _normalize(text)
    # Check in order: specific families first, then broader ones.
    # cars/gaming/hardware must come BEFORE health because:
    #   - gaming has "киберспорт" which contains "спорт" (a health term)
    #   - hardware has "ноут" which overlaps with local_business "ноутбук ремонт"
    #   - cars terms are distinct, but placed here for consistency
    # local_business must come before hardware to avoid "ремонт ноутбуков" → hardware.
    # NOTE: if you change this order, also update the fallback in news_service.py.
    priority_order = [
        "massage", "food", "beauty", "local_business",
        "cars", "gaming", "hardware",
        "health",
        "education", "finance", "marketing", "lifestyle", "expert_blog",
        "tech", "business",
    ]
    for family in priority_order:
        block = TOPIC_FAMILY_TERMS.get(family, {})
        terms = block.get("ru", []) + block.get("en", [])
        if any(token in q for token in terms):
            return family
    return "generic"


def get_family_image_queries(family: str) -> list[str]:
    """Returns stock-photo query templates for a given topic family."""
    block = TOPIC_FAMILY_TERMS.get(family) or {}
    queries = block.get("image_queries", [])
    if not queries:
        # Safe generic fallback — not tech-biased
        queries = [
            "professional editorial photo realistic",
            "workspace creative professional editorial photography",
        ]
    return queries


def get_family_allowed_visuals(family: str) -> list[str]:
    """Returns allowed visual classes for this family."""
    block = TOPIC_FAMILY_TERMS.get(family) or {}
    return block.get("allowed_visual_classes", [])


def get_family_blocked_visuals(family: str) -> list[str]:
    """Returns blocked visual classes for this family."""
    block = TOPIC_FAMILY_TERMS.get(family) or {}
    return block.get("blocked_visual_classes", [])


def get_family_news_angle(family: str) -> str:
    """Returns the news transformation angle for a given topic family."""
    block = TOPIC_FAMILY_TERMS.get(family) or {}
    return block.get("news_angle", (
        "Раскрой эту новость через призму темы канала. "
        "Что изменилось? Почему это важно для подписчиков прямо сейчас?"
    ))


def get_family_guardrails(family: str) -> str:
    """Returns the guardrail rules for text generation in a given family."""
    rules = FAMILY_GENERATION_RULES.get(family) or FAMILY_GENERATION_RULES["generic"]
    return rules.get("guardrails", FAMILY_GENERATION_RULES["generic"]["guardrails"])


def get_family_tone(family: str) -> str:
    """Returns tone descriptor for a given family."""
    rules = FAMILY_GENERATION_RULES.get(family) or FAMILY_GENERATION_RULES["generic"]
    return rules.get("tone", "concrete, human, practical")


def get_family_post_angles(family: str) -> list[str]:
    """Returns a list of useful post angles for the given family."""
    rules = FAMILY_GENERATION_RULES.get(family) or FAMILY_GENERATION_RULES["generic"]
    return rules.get("post_angles", [])


def get_family_cta_style(family: str) -> str:
    """Returns CTA style hints for the given family."""
    rules = FAMILY_GENERATION_RULES.get(family) or FAMILY_GENERATION_RULES["generic"]
    return rules.get("cta_style", "поделись мнением, попробуй, расскажи")


def classify_visual_type(topic: str = "", prompt: str = "", body: str = "") -> str:
    """Classify a post into a visual category for image selection policy."""
    src = _normalize(" ".join([topic or "", prompt or "", body or ""]))

    product_signals = [
        "продукт", "устройств", "гаджет", "обзор", "характеристик", "спецификац",
        "product", "device", "gadget", "review", "specification", "benchmark",
        "screenshot", "скриншот", "интерфейс", "ui", "ux", "приложен", "app",
    ]
    infra_signals = [
        "devops", "облак", "cloud", "сервер", "server", "kubernetes", "docker",
        "cicd", "инфраструктур", "infrastructure", "деплой", "deploy", "pipeline",
        "мониторинг", "monitoring", "масштабир", "scaling",
    ]
    analytics_signals = [
        "аналитик", "analytics", "метрик", "metric", "kpi", "тренд", "trend",
        "рынок", "market", "исследован", "research", "статистик", "statistic",
        "отчет", "report", "data", "данн", "roi", "конверс", "conversion",
    ]
    tutorial_signals = [
        "как ", "how to", "гайд", "guide", "пошаг", "step-by-step", "урок",
        "lesson", "туториал", "tutorial", "чеклист", "checklist", "инструкц",
        "instruction", "совет", "tip", "рекомендац",
    ]
    news_signals = [
        "новост", "news", "анонс", "announce", "запуск", "launch", "релиз",
        "release", "обновлен", "update", "презентац", "presentation",
    ]
    case_signals = [
        "кейс", "case study", "опыт", "experience", "пример", "example",
        "история успех", "success story", "практик", "practice",
    ]
    opinion_signals = [
        "мнение", "opinion", "размышлен", "reflection", "колонк", "column",
        "комментар", "commentary", "editorial", "дискуссия", "debate",
    ]
    text_only_signals = [
        "подборк", "список", "list", "цитат", "quote", "мем", "meme",
        "юмор", "humor", "анекдот", "joke", "афоризм", "aphorism",
    ]

    scores: dict[str, int] = {
        VISUAL_TYPE_PRODUCT: 0,
        VISUAL_TYPE_INFRASTRUCTURE: 0,
        VISUAL_TYPE_ANALYTICS: 0,
        VISUAL_TYPE_TUTORIAL: 0,
        VISUAL_TYPE_NEWS: 0,
        VISUAL_TYPE_EDITORIAL: 0,
        VISUAL_TYPE_CASE_STUDY: 0,
        VISUAL_TYPE_TEXT_ONLY: 0,
    }
    for s in product_signals:
        if s in src:
            scores[VISUAL_TYPE_PRODUCT] += 1
    for s in infra_signals:
        if s in src:
            scores[VISUAL_TYPE_INFRASTRUCTURE] += 1
    for s in analytics_signals:
        if s in src:
            scores[VISUAL_TYPE_ANALYTICS] += 1
    for s in tutorial_signals:
        if s in src:
            scores[VISUAL_TYPE_TUTORIAL] += 1
    for s in news_signals:
        if s in src:
            scores[VISUAL_TYPE_NEWS] += 1
    for s in case_signals:
        if s in src:
            scores[VISUAL_TYPE_CASE_STUDY] += 1
    for s in opinion_signals:
        if s in src:
            scores[VISUAL_TYPE_EDITORIAL] += 1
    for s in text_only_signals:
        if s in src:
            scores[VISUAL_TYPE_TEXT_ONLY] += 1

    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return VISUAL_TYPE_EDITORIAL
    return best


# ---------------------------------------------------------------------------
# Channel-aware image policy
# ---------------------------------------------------------------------------

# Families where image policy is strict: only semantically relevant images allowed.
# All known families are strict — we never want random off-topic images for any niche.
STRICT_IMAGE_FAMILIES: set[str] = {
    "tech", "business", "hardware", "generic",
    "food", "health", "beauty", "local_business",
    "education", "finance", "marketing",
    "massage", "cars", "gaming", "lifestyle", "expert_blog",
}

# Broad negative image categories to reject for strict-policy channels.
IRRELEVANT_IMAGE_CLASSES: list[str] = [
    "fashion", "model", "beauty", "glamour", "runway", "couture", "makeup",
    "lifestyle", "influencer", "selfie", "portrait studio",
    "animal", "pet", "cat", "dog", "puppy", "kitten", "wildlife",
    "romance", "wedding", "love", "couple kissing", "valentines",
    "food", "cooking", "recipe", "restaurant", "cuisine", "dish",
    "fitness", "gym", "workout", "bodybuilding", "yoga pose",
    "baby", "child portrait", "maternity",
    "tropical beach vacation", "resort", "travel selfie",
]


def get_family_irrelevant_classes(family: str) -> list[str]:
    """
    Returns irrelevant image classes for a given topic family.
    These are used to penalize/reject images that don't fit the channel niche.

    Combines family-specific blocked_visual_classes with the broad generic list,
    so both family-specific AND generic irrelevant classes are always penalized.
    """
    blocked = get_family_blocked_visuals(family)
    if blocked:
        # Merge family-specific + generic to catch both category-specific AND broad off-topic
        combined = set(blocked)
        combined.update(IRRELEVANT_IMAGE_CLASSES)
        return list(combined)
    return IRRELEVANT_IMAGE_CLASSES


# ---------------------------------------------------------------------------
# Subfamily detection — picks a more specific sub-niche within a family
# Used for building more precise image queries and scoring adjustments
# ---------------------------------------------------------------------------

_SUBFAMILY_RULES: dict[str, list[tuple[str, list[str]]]] = {
    "food": [
        ("coffee", ["кофе", "капучино", "латте", "эспрессо", "coffee", "cappuccino", "latte", "barista"]),
        ("bakery", ["выпечк", "хлеб", "тест", "кондитер", "булочн", "bakery", "bread", "pastry", "croissant"]),
        ("restaurant", ["ресторан", "кафе", "бистро", "ужин", "банкет", "restaurant", "cafe", "dining"]),
        ("recipe", ["рецепт", "готовить", "ингредиент", "recipe", "cooking", "homemade"]),
        ("nutrition", ["питани", "нутрицион", "калори", "диет", "nutrition", "diet", "calories"]),
    ],
    "health": [
        ("fitness", ["фитнес", "тренировк", "спорт", "зал", "упражнен", "fitness", "workout", "gym", "exercise"]),
        ("meditation", ["медитац", "йог", "дыхан", "осознан", "mindfulness", "meditation", "yoga", "breathing"]),
        ("nutrition", ["питани", "нутрицион", "еда здоров", "nutrition", "healthy food", "meal prep"]),
        ("clinic", ["врач", "клиник", "диагноз", "лечен", "симптом", "doctor", "clinic", "diagnosis", "treatment"]),
        ("mental", ["психолог", "ментальн", "стресс", "терапи", "psychology", "mental health", "stress", "therapy"]),
    ],
    "beauty": [
        ("nails", ["маникюр", "педикюр", "гель лак", "шеллак", "нейл", "nail", "manicure", "pedicure", "gel polish"]),
        ("hair", ["парикмахер", "стриж", "окраш", "укладк", "hair", "haircut", "coloring", "styling", "blowout"]),
        ("skincare", ["косметолог", "уход за кож", "сыворотк", "крем", "пилинг", "skincare", "serum", "cream", "peeling"]),
        ("clinic", ["инъекц", "ботокс", "филлер", "мезотерапи", "botox", "filler", "mesotherapy", "injection"]),
    ],
    "local_business": [
        ("repair", ["ремонт", "починк", "восстановлен", "repair", "fix"]),
        ("cleaning", ["клининг", "уборк", "cleaning", "janitorial"]),
        ("construction", ["строительств", "отделк", "ремонт квартир", "renovation", "construction"]),
        ("auto_service", ["автосервис", "шиномонтаж", "auto service", "tire"]),
    ],
    "cars": [
        ("interior", ["салон", "интерьер", "cockpit", "interior"]),
        ("engine", ["двигател", "engine", "мотор", "motor"]),
        ("service", ["ремонт", "сервис", "service", "repair", "обслуживан", "maintenance"]),
        ("detailing", ["детейлинг", "полировк", "химчистк", "detailing", "polish"]),
        ("tire", ["шин", "колес", "tire", "wheel"]),
    ],
    "finance": [
        ("crypto", ["крипт", "биткоин", "эфириум", "блокчейн", "crypto", "bitcoin", "ethereum", "blockchain"]),
        ("analytics", ["аналитик", "график", "chart", "analytics", "data"]),
        ("planning", ["планиров", "бюджет", "financial planning", "budget"]),
        ("trading", ["трейдинг", "биржа", "forex", "trading", "exchange"]),
    ],
    "marketing": [
        ("smm", ["smm", "соцсет", "социальн сет", "social media"]),
        ("analytics", ["аналитик", "метрик", "kpi", "analytics", "metrics"]),
        ("content", ["контент", "копирайтинг", "тексты", "content", "copywriting"]),
    ],
    "education": [
        ("online", ["онлайн", "курс", "дистанц", "online", "course", "distance"]),
        ("books", ["книг", "библиотек", "book", "library", "reading"]),
        ("training", ["тренинг", "вебинар", "воркшоп", "training", "webinar", "workshop"]),
    ],
    "massage": [
        ("neck", ["шея", "шеи", "плеч", "neck", "shoulder"]),
        ("back", ["спина", "спины", "поясниц", "back", "lower back"]),
        ("face", ["лицо", "лица", "face", "jaw"]),
    ],
    "tech": [
        ("ai", ["ai", "ии", "нейросет", "gpt", "llm", "machine learning", "deep learning"]),
        ("devops", ["devops", "облак", "cloud", "сервер", "docker", "kubernetes"]),
        ("coding", ["код", "программ", "developer", "coding", "python", "javascript"]),
    ],
}

# Subfamily-specific image query templates — override family defaults when subfamily is detected
_SUBFAMILY_IMAGE_QUERIES: dict[str, dict[str, list[str]]] = {
    "food": {
        "coffee": [
            "coffee cup cafe barista editorial photography",
            "coffee shop morning mood beans cup realistic photo",
            "latte art cafe interior professional editorial",
        ],
        "bakery": [
            "bakery bread pastry fresh baked editorial photography",
            "croissant bread basket bakery shop editorial photo",
            "homemade baking kitchen fresh dough editorial",
        ],
        "restaurant": [
            "restaurant dish gourmet plating professional editorial photo",
            "dining table food atmosphere restaurant realistic photo",
            "fine dining plate presentation editorial",
        ],
        "recipe": [
            "kitchen cooking ingredients flat lay editorial photography",
            "homemade food preparation close-up natural light photo",
            "recipe ingredients chopping board kitchen editorial",
        ],
        "nutrition": [
            "healthy food nutrition meal prep clean editorial",
            "organic vegetables fruits nutrition photo editorial",
            "balanced meal portion healthy plate editorial",
        ],
    },
    "health": {
        "fitness": [
            "fitness workout exercise healthy lifestyle realistic editorial photo",
            "sport training motivation gym realistic photo",
            "athlete exercise equipment professional editorial",
        ],
        "meditation": [
            "meditation mindfulness peaceful nature editorial",
            "yoga practice calm peaceful studio editorial photo",
            "breathing exercise relaxation nature editorial",
        ],
        "clinic": [
            "medical professional consultation warm editorial photo",
            "doctor patient consultation clinic editorial",
            "healthcare professional clinic interior editorial",
        ],
        "mental": [
            "mental health therapy calm nature editorial photo",
            "psychology session comfortable room editorial",
            "stress relief calm nature peaceful editorial",
        ],
    },
    "beauty": {
        "nails": [
            "nail art manicure creative beauty editorial",
            "gel polish nail studio professional editorial photo",
            "manicure process salon hands close-up editorial",
        ],
        "hair": [
            "hair styling beauty salon professional editorial",
            "haircut coloring stylist salon editorial photo",
            "hairdresser blow-dry styling editorial",
        ],
        "skincare": [
            "skincare serum cream beauty product editorial photo",
            "facial treatment skincare routine close-up editorial",
            "beauty product flat lay skincare routine editorial",
        ],
        "clinic": [
            "cosmetology clinic treatment professional editorial",
            "beauty clinic procedure professional editorial photo",
            "aesthetics clinic professional treatment editorial",
        ],
    },
    "cars": {
        "interior": [
            "car interior dashboard steering wheel editorial photo",
            "luxury car interior cockpit detail editorial",
            "vehicle cabin dashboard controls editorial",
        ],
        "engine": [
            "car engine repair mechanic garage editorial photo",
            "engine compartment automotive detail editorial",
            "auto mechanic engine repair workshop editorial",
        ],
        "service": [
            "car service repair garage mechanic editorial photo",
            "automotive workshop tools mechanic editorial",
            "car maintenance service center professional editorial",
        ],
        "detailing": [
            "car detailing polishing professional editorial photo",
            "vehicle detailing wash clean editorial",
            "auto detailing interior cleaning editorial",
        ],
        "tire": [
            "tire change service garage editorial photo",
            "wheel tire automotive service editorial",
            "tire shop wheel alignment editorial",
        ],
    },
    "massage": {
        "neck": [
            "neck shoulder massage therapy hands editorial photo",
            "cervical massage therapist hands neck editorial",
            "shoulder massage treatment professional editorial",
        ],
        "back": [
            "back massage therapy hands editorial photo",
            "lower back treatment therapist editorial",
            "spinal massage recovery treatment editorial",
        ],
        "face": [
            "facial massage therapy hands face editorial photo",
            "face massage relaxation spa editorial",
            "jaw facial massage treatment editorial",
        ],
    },
}

# Per-family URL signal rejection lists — specific terms that indicate off-topic
# images when found in URLs for each family
FAMILY_BAD_URL_SIGNALS: dict[str, list[str]] = {
    "food": ["circuit", "server", "code", "corporate", "finance", "abstract", "neon", "gadget", "gaming"],
    "health": ["fashion", "luxury", "glamour", "circuit", "gaming", "stock chart", "abstract", "neon", "food", "restaurant", "cuisine", "recipe", "cooking", "dish"],
    "beauty": ["circuit", "server", "gaming", "finance", "abstract neon", "corporate", "car", "engine"],
    "local_business": ["abstract", "neon", "cartoon", "fashion", "gaming", "glamour", "circuit"],
    "education": ["fashion", "glamour", "gaming", "abstract neon", "luxury", "beauty model"],
    "finance": ["fashion", "food", "beauty", "gaming", "cartoon", "pet", "animal", "abstract neon"],
    "marketing": ["cartoon", "gaming", "abstract neon", "food", "animal", "pet"],
    "lifestyle": ["server", "circuit", "abstract tech", "gaming setup", "corporate"],
    "expert_blog": ["cartoon", "abstract neon", "gaming", "food flat lay", "beauty model"],
    "massage": ["illustration", "cartoon", "spa candles", "cream", "beauty salon", "fashion", "tech", "food", "restaurant", "cuisine", "recipe", "cooking", "dish"],
    "hardware": ["3d render", "mockup", "cartoon", "rgb wallpaper", "fashion", "food", "beauty"],
    "gaming": ["poster", "cover art", "fan art", "mascot", "fashion", "food", "beauty"],
    "cars": ["logo", "render", "toy", "cartoon", "fashion", "food", "beauty", "abstract", "watch"],
    "tech": ["fashion", "food", "beauty", "cartoon", "animal", "romance"],
    "business": ["cartoon", "food", "beauty", "gaming", "animal", "fashion"],
    "generic": ["fashion", "model", "beauty", "glamour", "wedding", "animal", "pet", "food", "recipe", "fitness", "baby", "romance"],
}


def detect_subfamily(family: str, text: str) -> str:
    """Detect a subfamily within a given topic family based on text signals.

    Returns the subfamily name (e.g. 'coffee', 'nails', 'neck') or empty string.
    """
    rules = _SUBFAMILY_RULES.get(family)
    if not rules:
        return ""
    q = _normalize(text)
    for name, tokens in rules:
        if any(token in q for token in tokens):
            return name
    return ""


def get_subfamily_image_queries(family: str, subfamily: str) -> list[str]:
    """Return image queries for a specific subfamily, or empty list if none defined."""
    return _SUBFAMILY_IMAGE_QUERIES.get(family, {}).get(subfamily, [])


def get_family_bad_url_signals(family: str) -> list[str]:
    """Return URL signal terms that indicate off-topic images for a given family."""
    return FAMILY_BAD_URL_SIGNALS.get(family, FAMILY_BAD_URL_SIGNALS["generic"])


__all__ = [
    "TOPIC_FAMILY_TERMS",
    "FAMILY_GENERATION_RULES",
    "STRICT_IMAGE_FAMILIES",
    "IRRELEVANT_IMAGE_CLASSES",
    "FAMILY_BAD_URL_SIGNALS",
    "detect_topic_family",
    "detect_subfamily",
    "get_family_image_queries",
    "get_subfamily_image_queries",
    "get_family_allowed_visuals",
    "get_family_blocked_visuals",
    "get_family_news_angle",
    "get_family_guardrails",
    "get_family_tone",
    "get_family_post_angles",
    "get_family_cta_style",
    "get_family_irrelevant_classes",
    "get_family_bad_url_signals",
    "classify_visual_type",
    "VISUAL_TYPE_PRODUCT",
    "VISUAL_TYPE_INFRASTRUCTURE",
    "VISUAL_TYPE_ANALYTICS",
    "VISUAL_TYPE_TUTORIAL",
    "VISUAL_TYPE_NEWS",
    "VISUAL_TYPE_EDITORIAL",
    "VISUAL_TYPE_CASE_STUDY",
    "VISUAL_TYPE_TEXT_ONLY",
]
