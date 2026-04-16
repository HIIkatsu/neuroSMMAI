from __future__ import annotations

from dataclasses import dataclass

from visual_profile_layer import ProviderCandidate, build_visual_profile, profile_search_queries, score_candidate


@dataclass
class Case:
    title: str
    channel_topic: str
    domain: str
    expected_outcome: str  # relevant | no_image
    candidates: list[ProviderCandidate]
    body: str = ""
    text_quality_flagged: bool = False
    problematic: bool = False


def make_candidate(domain_tag: str, *, good: bool, provider: str = "pexels") -> ProviderCandidate:
    if good:
        return ProviderCandidate(
            url=f"https://images.example.com/{domain_tag}/good.jpg",
            provider=provider,
            caption=f"{domain_tag} editorial real-life scene",
            tags=[domain_tag, "editorial", "realistic"],
            author="photographer",
            width=1920,
            height=1280,
        )
    return ProviderCandidate(
        url=f"https://images.example.com/{domain_tag}/bad.jpg",
        provider=provider,
        caption="generic office stock photo handshake meeting",
        tags=["generic office", "meeting", "stock photo"],
        author="stock",
        width=1600,
        height=900,
    )


def classify_result(expected_domain: str, picked: ProviderCandidate | None) -> str:
    if not picked:
        return "no_image"
    text = f"{picked.caption} {' '.join(picked.tags)}".lower()
    return "relevant" if expected_domain in text else "wrong_image"


def legacy_pick(candidates: list[ProviderCandidate]) -> ProviderCandidate | None:
    return candidates[0] if candidates else None


def current_pick(case: Case):
    profile = build_visual_profile(
        title=case.title,
        channel_topic=case.channel_topic,
        body="" if case.text_quality_flagged else case.body,
        onboarding_summary=f"rubric {case.channel_topic}",
        post_intent="educational post",
    )
    query, _ = profile_search_queries(profile)

    best = None
    best_breakdown = None
    best_score = -10**9
    for cand in case.candidates:
        cand.source_query = query
        breakdown = score_candidate(candidate=cand, profile=profile, min_score=1.5)
        if breakdown.score > best_score:
            best_score = breakdown.score
            best = cand
            best_breakdown = breakdown

    if not best_breakdown or best_breakdown.decision != "accepted":
        return profile, query, None, "no_image", best_breakdown.reason if best_breakdown else "no_candidates"
    return profile, query, best, "accepted", best_breakdown.reason


CASES: list[Case] = [
    # problematic (from old regressions/log-like titles)
    Case("Вклады в банке и проценты", "finance", "finance", "relevant", [make_candidate("finance", good=False), make_candidate("finance", good=True)], text_quality_flagged=True, body="ROI 200% turbo", problematic=True),
    Case("Как выбрать правильный самокат для города", "scooter", "scooter", "relevant", [make_candidate("cars", good=False), make_candidate("scooter", good=True)], problematic=True),
    Case("Агрономия: почва, семена и урожай", "gardening", "gardening", "relevant", [make_candidate("finance", good=False), make_candidate("gardening", good=True)], problematic=True),
    Case("NFT-инвесторы теряют деньги: анализ рисков", "finance", "finance", "relevant", [make_candidate("tech", good=False), make_candidate("finance", good=True)], problematic=True),
    Case("Технический обзор серверов", "tech", "electronics", "relevant", [make_candidate("local_news", good=False), make_candidate("electronics", good=True)], problematic=True),
    Case("Вклад в банке: как выбрать депозит", "finance", "finance", "relevant", [make_candidate("cars", good=False), make_candidate("finance", good=True)], problematic=True),
    Case("Почва и семена: старт сезона", "gardening", "gardening", "relevant", [make_candidate("finance", good=False), make_candidate("gardening", good=True)], problematic=True),
    Case("Город открыл новую автобусную полосу", "local_news", "local_news", "relevant", [make_candidate("electronics", good=False), make_candidate("local_news", good=True)], problematic=True),
    Case("Как выбрать семейного врача", "health", "health", "relevant", [make_candidate("finance", good=False), make_candidate("health", good=True)], problematic=True),
    Case("Технический обзор серверов для дата-центра", "tech", "electronics", "relevant", [make_candidate("cars", good=False), make_candidate("electronics", good=True)], problematic=True),

    # additional live-like mix across required domains
    Case("Рост ставки по ипотеке в городе", "finance", "finance", "relevant", [make_candidate("finance", good=True), make_candidate("local_news", good=False)]),
    Case("Как выбрать дебетовую карту без комиссий", "finance", "finance", "relevant", [make_candidate("finance", good=True)]),
    Case("Замена тормозных колодок перед зимой", "cars", "cars", "relevant", [make_candidate("cars", good=True), make_candidate("finance", good=False)]),
    Case("Перегрев двигателя в пробке", "cars", "cars", "relevant", [make_candidate("cars", good=True)]),
    Case("Электросамокат: почему скрипит подвеска", "scooter", "scooter", "relevant", [make_candidate("scooter", good=True)]),
    Case("Безопасная езда на самокате ночью", "scooter", "scooter", "relevant", [make_candidate("scooter", good=True), make_candidate("health", good=False)]),
    Case("Клиника: подготовка к ежегодному чекапу", "health", "health", "relevant", [make_candidate("health", good=True)]),
    Case("Реабилитация после травмы колена", "health", "health", "relevant", [make_candidate("health", good=True)]),
    Case("Муниципальное уведомление о воде", "local_news", "local_news", "relevant", [make_candidate("local_news", good=True), make_candidate("finance", good=False)]),
    Case("Когда пересаживать рассаду томатов", "gardening", "gardening", "relevant", [make_candidate("gardening", good=True)]),
    Case("Как измерять pH почвы", "gardening", "gardening", "relevant", [make_candidate("gardening", good=True), make_candidate("electronics", good=False)]),
    Case("Как готовиться к экзамену без выгорания", "education", "education", "relevant", [make_candidate("education", good=True)]),
    Case("Конспектирование лекций онлайн", "education", "education", "relevant", [make_candidate("education", good=True), make_candidate("tech", good=False)]),
    Case("Как продлить срок службы батареи ноутбука", "tech", "electronics", "relevant", [make_candidate("electronics", good=True), make_candidate("cars", good=False)]),
    Case("Сравнение камер смартфонов", "tech", "electronics", "relevant", [make_candidate("electronics", good=True)]),
    Case("Уход за чувствительной кожей утром", "beauty", "beauty", "relevant", [make_candidate("beauty", good=True), make_candidate("health", good=False)]),
    Case("График вакцинации щенка", "pets", "pets", "relevant", [make_candidate("pets", good=True)]),
    Case("Когда кошке срочно нужен ветеринар", "pets", "pets", "relevant", [make_candidate("pets", good=True)]),
    Case("Очень абстрактный финансовый пост без объекта", "finance", "finance", "no_image", [make_candidate("finance", good=False)]),
    Case("Служебная заметка города без визуального контекста", "local_news", "local_news", "no_image", [make_candidate("local_news", good=False)]),
]


if __name__ == "__main__":
    rows = []
    wrong_count = 0
    no_image_count = 0

    for case in CASES:
        profile, query, picked, decision, reason = current_pick(case)
        final = classify_result(case.domain, picked)
        if final == "wrong_image":
            wrong_count += 1
        if final == "no_image":
            no_image_count += 1
        rows.append((case, profile, query, picked, decision, reason, final))

    print(f"TOTAL={len(rows)} WRONG={wrong_count} NO_IMAGE={no_image_count}")
    print("| title | canonical profile / primary subject | search query | provider | candidate caption/tags | final decision | accepted/rejected reason | итог |")
    print("|---|---|---|---|---|---|---|---|")
    for case, profile, query, picked, decision, reason, final in rows:
        caption_tags = "-" if not picked else f"{picked.caption} / {','.join(picked.tags)}"
        provider = "-" if not picked else picked.provider
        print(f"| {case.title} | {profile.domain_family} / {profile.primary_subject[:48]} | {query[:72]} | {provider} | {caption_tags[:84]} | {decision} | {reason} | {final} |")

    print("\n| problematic title | before (legacy first-valid) | after (current scoring) |")
    print("|---|---|---|")
    for case in CASES:
        if not case.problematic:
            continue
        before = classify_result(case.domain, legacy_pick(case.candidates))
        _, _, after_pick, _, _ = current_pick(case)
        after = classify_result(case.domain, after_pick)
        print(f"| {case.title} | {before} | {after} |")
