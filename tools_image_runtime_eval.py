from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from visual_profile_layer import ProviderCandidate, build_visual_profile, profile_search_queries, score_candidate


@dataclass
class Case:
    title: str
    channel_topic: str
    expected_domain: str
    expected_outcome: str  # relevant | no_image
    body: str
    candidates: list[ProviderCandidate]


def pexels(url_slug: str, caption: str, tags: list[str]) -> ProviderCandidate:
    return ProviderCandidate(
        url=f"https://images.pexels.com/photos/{url_slug}.jpeg",
        provider="pexels",
        caption=caption,
        tags=tags,
        source_query="",
    )


def pixabay(url_slug: str, caption: str, tags: list[str]) -> ProviderCandidate:
    return ProviderCandidate(
        url=f"https://cdn.pixabay.com/photo/{url_slug}.jpg",
        provider="pixabay",
        caption=caption,
        tags=tags,
        source_query="",
    )


def make_pair(domain: str, subject: str, wrong_hint: str = "generic office business handshake") -> list[ProviderCandidate]:
    return [
        pexels(f"bad-{domain}", wrong_hint, ["generic office", "business team", "meeting"]),
        pixabay(f"good-{domain}", f"{subject} real scene", [domain, subject.split()[0], "street"]),
    ]


CASES: list[Case] = [
    # finance
    Case("Вклад в банке: как выбрать депозит", "finance", "finance", "relevant", "Подбор ставки и рисков", make_pair("finance", "banking consultation")),
    Case("Банк запустил карту для студентов", "finance", "finance", "relevant", "Новые лимиты и кешбэк", make_pair("finance", "banking consultation")),
    Case("Рефинансирование ипотеки: чеклист", "finance", "finance", "relevant", "Когда выгодно рефинансировать", make_pair("finance", "banking consultation")),
    Case("Почему инфляция съедает накопления", "finance", "finance", "relevant", "Личный бюджет семьи", make_pair("finance", "banking consultation")),

    # local_news
    Case("Муниципалитет открыл новый МФЦ", "local_news", "local_news", "relevant", "График и услуги центра", make_pair("local_news", "municipal service")),
    Case("В районе отключат воду на сутки", "local_news", "local_news", "relevant", "Адреса и время работ", make_pair("local_news", "municipal service")),
    Case("Город обновляет уличное освещение", "local_news", "local_news", "relevant", "Сроки по кварталам", make_pair("local_news", "municipal service")),
    Case("Слушания по реконструкции площади", "local_news", "local_news", "relevant", "Публичная встреча жителей", make_pair("local_news", "municipal service")),

    # transport
    Case("Новый автобусный маршрут до вокзала", "transport", "transport", "relevant", "Схема остановок", make_pair("transport", "public transport")),
    Case("Метро закроет станцию на ремонт", "transport", "transport", "relevant", "Альтернативные пути", make_pair("transport", "public transport")),
    Case("Трамвай продлили до больницы", "transport", "transport", "relevant", "Первый рейс в понедельник", make_pair("transport", "public transport")),
    Case("Пробки вырастут из-за ремонта моста", "transport", "transport", "relevant", "Ограничения полос", make_pair("transport", "public transport")),

    # health
    Case("Как выбрать семейного врача", "health", "health", "relevant", "Критерии и вопросы на приёме", make_pair("health", "medical consultation")),
    Case("Поликлиника запустила онлайн-повтор", "health", "health", "relevant", "Повторные консультации", make_pair("health", "medical consultation")),
    Case("Годовой чекап: какие анализы сдать", "health", "health", "relevant", "Профилактика для взрослых", make_pair("health", "medical consultation")),
    Case("Реабилитация после травмы колена", "health", "health", "relevant", "Этапы восстановления", make_pair("health", "medical consultation")),

    # real_estate
    Case("Проверка квартиры перед покупкой", "real_estate", "real_estate", "relevant", "Список технических пунктов", make_pair("real_estate", "property viewing")),
    Case("Арендный договор: красные флаги", "real_estate", "real_estate", "relevant", "Штрафы и депозиты", make_pair("real_estate", "property viewing")),
    Case("Спрос на студии вырос в центре", "real_estate", "real_estate", "relevant", "Статистика по районам", make_pair("real_estate", "property viewing")),
    Case("Ипотека для первичного жилья", "real_estate", "real_estate", "relevant", "Требования банков", make_pair("real_estate", "property viewing")),

    # food
    Case("Ресторан добавил постное меню", "food", "food", "relevant", "Новые позиции недели", make_pair("food", "food preparation")),
    Case("Как готовить суп на 3 дня", "food", "food", "relevant", "Meal prep для семьи", make_pair("food", "food preparation")),
    Case("Школьные столовые тестируют меню", "food", "food", "relevant", "Баланс БЖУ", make_pair("food", "food preparation")),
    Case("Снижение цен на овощи весной", "food", "food", "relevant", "Рынок поставок", make_pair("food", "food preparation")),

    # services
    Case("Сервис-центр ввёл запись день-в-день", "services", "services", "relevant", "Время ожидания сократилось", make_pair("services", "service workflow")),
    Case("Клининговые компании усилили обучение", "services", "services", "relevant", "Стандарты качества", make_pair("services", "service workflow")),
    Case("Курьерская служба расширила вечернюю доставку", "services", "services", "relevant", "Новые окна доставки", make_pair("services", "service workflow")),
    Case("Как сравнить сметы подрядчиков", "services", "services", "relevant", "На что смотреть в договоре", make_pair("services", "service workflow")),

    # lifestyle
    Case("Утренняя рутина и концентрация", "lifestyle", "lifestyle", "relevant", "Привычки на рабочий день", make_pair("lifestyle", "daily routine")),
    Case("Как восстановить режим сна", "lifestyle", "lifestyle", "relevant", "После перелёта", make_pair("lifestyle", "daily routine")),
    Case("Минимализм в маленькой квартире", "lifestyle", "lifestyle", "relevant", "Организация пространства", make_pair("lifestyle", "daily routine")),
    Case("Цифровой детокс на выходных", "lifestyle", "lifestyle", "relevant", "План на 48 часов", make_pair("lifestyle", "daily routine")),

    # education
    Case("Школы добавили кружки программирования", "education", "education", "relevant", "Набор на семестр", make_pair("education", "learning activity")),
    Case("Как готовиться к экзамену без выгорания", "education", "education", "relevant", "План повторения", make_pair("education", "learning activity")),
    Case("Университет обновил правила поступления", "education", "education", "relevant", "Новые дедлайны", make_pair("education", "learning activity")),
    Case("Учителя тестируют политику ИИ", "education", "education", "relevant", "Классные практики", make_pair("education", "learning activity")),

    # auto
    Case("Когда менять тормозные колодки", "cars", "cars", "relevant", "Признаки износа", make_pair("cars", "car service")),
    Case("Подготовка авто к зиме", "cars", "cars", "relevant", "Проверка АКБ и шин", make_pair("cars", "car service")),
    Case("Перегрев двигателя в пробке", "cars", "cars", "relevant", "Причины и диагностика", make_pair("cars", "car service")),
    Case("Проверка б/у автомобиля перед покупкой", "cars", "cars", "relevant", "Чеклист диагностики", make_pair("cars", "car service")),

    # intentionally hard/bad (for top-5 remaining issues)
    Case("Апдейт", "local_news", "local_news", "relevant", "Короткая служебная заметка без объекта", [pexels("bad-amb-1", "generic office business handshake", ["generic office", "business"])]),
    Case("Сервис", "services", "services", "no_image", "Очень короткий заголовок без сцены", [pexels("bad-amb-2", "team meeting office", ["office", "meeting"])]),
    Case("Рынок", "finance", "finance", "no_image", "Слишком абстрактно без предмета", [pexels("bad-amb-3", "abstract graph office", ["graph", "office"])]),
    Case("Курс", "education", "education", "relevant", "Нет визуального предмета", [pexels("bad-amb-4", "generic office workshop", ["office", "workshop"])]),
]


SHORT_PROMPT_CASES = [
    ("мусор", "РЕЦЕПТЫ и office hardware meeting", "services"),
    ("банк", "GPU hardware datacenter mining", "finance"),
    ("самокат", "corporate office team and business", "scooter"),
    ("тормоза", "news local office update", "cars"),
    ("школа", "servers and hardware benchmark", "education"),
]


def classify(case: Case, picked: ProviderCandidate | None) -> str:
    if picked is None:
        return "no_image"
    text = f"{picked.caption} {' '.join(picked.tags)}".lower()
    if "generic office" in text or "business handshake" in text:
        return "wrong_image"
    expected = case.expected_domain.lower()
    return "relevant" if expected in text else "wrong_image"


def legacy_before_pick(candidates: list[ProviderCandidate]) -> ProviderCandidate | None:
    # Legacy-like behavior: first valid result from provider, no strict intent scoring.
    return candidates[0] if candidates else None


def runtime_after_pick(case: Case):
    profile = build_visual_profile(
        title=case.title,
        body=case.body,
        channel_topic=case.channel_topic,
        onboarding_summary=f"rubric for {case.channel_topic}",
        post_intent="manual editor intent",
        subniche=case.expected_domain,
    )
    primary_q, backup_q = profile_search_queries(profile)
    best: ProviderCandidate | None = None
    best_reason = "no_candidates"
    best_score = -10**9
    for cand in case.candidates:
        cand.source_query = primary_q
        s = score_candidate(candidate=cand, profile=profile, min_score=2.0)
        if s.score > best_score:
            best_score = s.score
            best = cand
            best_reason = s.reason
    if best is None:
        return profile, primary_q, backup_q, None, "no_candidates"
    final_s = score_candidate(candidate=best, profile=profile, min_score=2.0)
    if final_s.decision != "accepted":
        return profile, primary_q, backup_q, None, final_s.reason
    return profile, primary_q, backup_q, best, final_s.reason


def evaluate() -> str:
    rows: list[str] = []
    before = {"relevant": 0, "wrong_image": 0, "no_image": 0}
    after = {"relevant": 0, "wrong_image": 0, "no_image": 0}
    still_bad: list[tuple[str, str, str]] = []

    rows.append("| title | detected domain | primary_subject | final primary query | backup query | provider | top candidate url/caption/tags | final decision | reject reason |")
    rows.append("|---|---|---|---|---|---|---|---|---|")

    for case in CASES:
        before_pick = legacy_before_pick(case.candidates)
        before_decision = classify(case, before_pick)
        before[before_decision] += 1

        profile, primary_q, backup_q, after_pick, reason = runtime_after_pick(case)
        after_decision = classify(case, after_pick)
        after[after_decision] += 1
        if case.expected_outcome == "relevant" and after_decision != "relevant":
            still_bad.append((case.title, after_decision, reason))
        if case.expected_outcome == "no_image" and after_decision != "no_image":
            still_bad.append((case.title, after_decision, reason))

        provider = after_pick.provider if after_pick else "-"
        cand_info = "-"
        if after_pick:
            cand_info = f"{after_pick.url} / {after_pick.caption} / {','.join(after_pick.tags)}"
        rows.append(
            f"| {case.title} | {profile.domain_family} | {profile.primary_subject} | {primary_q} | {backup_q} | {provider} | {cand_info} | {after_decision} | {'' if after_pick else reason} |"
        )

    rows.append("")
    rows.append("## Before/After (runtime replay)")
    rows.append(f"- before: relevant={before['relevant']}, wrong_image={before['wrong_image']}, no_image={before['no_image']}")
    rows.append(f"- after: relevant={after['relevant']}, wrong_image={after['wrong_image']}, no_image={after['no_image']}")

    rows.append("")
    rows.append("## Editor/manual short prompt = user input law")
    rows.append("| prompt | noisy body | detected domain | primary query | drift check |")
    rows.append("|---|---|---|---|---|")
    for prompt, noisy_body, expected_domain in SHORT_PROMPT_CASES:
        profile = build_visual_profile(
            title=prompt,
            body=noisy_body,
            channel_topic="",
            onboarding_summary="",
            post_intent=prompt,
            subniche=expected_domain,
            text_quality_flagged=False,
        )
        q1, _ = profile_search_queries(profile)
        alias = {"cars": {"cars", "auto"}}
        accepted = alias.get(expected_domain, {expected_domain})
        drift = "ok" if profile.domain_family in accepted else "drift"
        rows.append(f"| {prompt} | {noisy_body} | {profile.domain_family} | {q1} | {drift} |")

    rows.append("")
    rows.append("## 5 still-bad cases")
    if not still_bad:
        rows.append("- none")
    else:
        for title, final_decision, reason in still_bad[:5]:
            rows.append(f"- {title}: {final_decision} ({reason})")

    return "\n".join(rows)


if __name__ == "__main__":
    report = evaluate()
    out = Path("RUNTIME_IMAGE_REPLAY_REPORT.md")
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved: {out}")
