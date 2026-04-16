from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Iterable
from unittest.mock import AsyncMock, patch

from image_history import ImageHistory
from image_service import (
    MODE_AUTOPOST,
    MODE_EDITOR,
    _build_simple_search_query,
    _is_simple_candidate_ok,
    get_image,
)
from visual_profile_layer import ProviderCandidate


@dataclass(frozen=True)
class ReplayCase:
    case_id: str
    mode: str
    explicit_query: str
    title: str
    body: str
    channel_topic: str
    expected_tokens: tuple[str, ...]


def _cases() -> list[ReplayCase]:
    return [
        ReplayCase("c01", MODE_EDITOR, "мусор", "Экология города", "Текст про офис и community", "локальные новости", ("мусор",)),
        ReplayCase("c02", MODE_EDITOR, "банк комиссии", "Local office update", "hardware and startup digest", "технологии", ("банк", "комис")),
        ReplayCase("c03", MODE_EDITOR, "вклад", "Обзор ресторана", "новости заведения", "еда", ("вклад",)),
        ReplayCase("c04", MODE_EDITOR, "больница", "Утилизация бумаги", "мусор и переработка", "local", ("больниц",)),
        ReplayCase("c05", MODE_EDITOR, "поликлиника", "Городской бюджет", "про налоги", "финансы", ("поликлиник",)),
        ReplayCase("c06", MODE_AUTOPOST, "", "Школа открыла новый класс", "Учителя проводят занятия", "город", ("школ", "класс", "учител")),
        ReplayCase("c07", MODE_AUTOPOST, "", "Трамвайный маршрут продлили", "Новый трамвай вышел на линию", "транспорт", ("трамва",)),
        ReplayCase("c08", MODE_AUTOPOST, "", "Рынок недвижимости оживился", "Квартиры и аренда дорожают", "новости", ("кварт", "недвиж", "аренд")),
        ReplayCase("c09", MODE_AUTOPOST, "", "Открылся новый ресторан", "Заведение готовит сезонное меню", "еда", ("ресторан", "заведен")),
        ReplayCase("c10", MODE_AUTOPOST, "", "Сервис для Жигули", "Авто ремонт и диагностика", "авто", ("авто", "жигул", "сервис")),
        ReplayCase("c11", MODE_AUTOPOST, "", "Банк снизил комиссии", "Условия вкладов обновлены", "финансы", ("банк", "комисс", "вклад")),
        ReplayCase("c12", MODE_AUTOPOST, "", "Поликлиника получила оборудование", "Больница расширяет приём", "здоровье", ("поликлиник", "больниц")),
        ReplayCase("c13", MODE_AUTOPOST, "", "Аренда квартир растёт", "Недвижимость в центре", "рынок", ("аренд", "кварт", "недвиж")),
        ReplayCase("c14", MODE_AUTOPOST, "", "Транспортный департамент купил трамвай", "Маршрут обновили", "транспорт", ("трамва", "транспорт")),
        ReplayCase("c15", MODE_EDITOR, "аптека", "Дорожные работы", "стройка и подрядчики", "город", ("аптек",)),
        ReplayCase("c16", MODE_EDITOR, "автосервис жигули", "Ресторан недели", "food digest", "еда", ("автосервис", "жигул")),
        ReplayCase("c17", MODE_AUTOPOST, "", "Школьный учитель победил конкурс", "Класс поддержал проект", "образование", ("школь", "учител", "класс")),
        ReplayCase("c18", MODE_AUTOPOST, "", "Вывоз мусора подорожал", "Контейнеры и переработка", "город", ("мусор", "вывоз")),
        ReplayCase("c19", MODE_AUTOPOST, "", "Ресторан у вокзала", "Заведение расширяет зал", "бизнес", ("ресторан", "заведен")),
        ReplayCase("c20", MODE_AUTOPOST, "", "Банк запустил вклад", "Комиссии стали ниже", "финансы", ("банк", "вклад", "комисс")),
    ]


def _candidate_pool(case: ReplayCase, final_query: str) -> list[ProviderCandidate]:
    q = final_query.replace(" ", "-")
    good = ProviderCandidate(
        url=f"https://images.pexels.com/photos/{case.case_id}-{q}.jpg",
        provider="pexels",
        caption=f"{final_query} street photo",
        tags=final_query.split(),
    )
    wrong = ProviderCandidate(
        url=f"https://cdn.example.com/wrong-office-{case.case_id}.jpg",
        provider="pixabay",
        caption="generic office meeting",
        tags=["office", "meeting"],
    )
    text_heavy = ProviderCandidate(
        url=f"https://cdn.example.com/poster-watermark-{case.case_id}.jpg",
        provider="pixabay",
        caption="poster with big text watermark",
        tags=["poster", "watermark", "text"],
    )

    # One hard case: return only irrelevant/text-heavy candidates to force honest no_image.
    if case.case_id == "c18":
        return [wrong, text_heavy, wrong]

    return [good, wrong, text_heavy]


def _outcome(case: ReplayCase, chosen_url: str) -> str:
    if not chosen_url:
        return "no_image"
    low = chosen_url.lower()
    if any(tok in low for tok in case.expected_tokens):
        return "relevant"
    return "wrong_image"


async def run_replay(*, emit: bool = True, enforce_gate: bool = False) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    wrong = 0
    no_image = 0

    for case in _cases():
        final_query = _build_simple_search_query(
            mode=case.mode,
            explicit_query=case.explicit_query,
            title=case.title,
            body=case.body,
        )
        candidates = _candidate_pool(case, final_query)
        chosen_provider = "none"

        with patch("image_service.search_stock_candidates", new_callable=AsyncMock) as mock_search, patch(
            "image_service.generate_image", new_callable=AsyncMock
        ) as mock_gen, patch("image_service.get_image_history", return_value=ImageHistory(maxlen=20, ttl=3600)):
            mock_search.return_value = candidates
            mock_gen.return_value = None
            result = await get_image(
                mode=case.mode,
                llm_image_prompt=case.explicit_query,
                title=case.title,
                body=case.body,
                channel_topic=case.channel_topic,
                api_key="test",
                model="mistral-small",  # text model, generation intentionally skipped
            )

        if result.media_ref:
            for c in candidates:
                if c.url == result.media_ref:
                    chosen_provider = c.provider
                    break

        outcome = _outcome(case, result.media_ref)
        if outcome == "wrong_image":
            wrong += 1
        if outcome == "no_image":
            no_image += 1

        rejected_reason = ""
        if not result.media_ref:
            reasons: list[str] = []
            for c in candidates[:3]:
                ok, reason = _is_simple_candidate_ok(c, query=final_query, used_refs=None)
                reasons.append(reason if not ok else "accepted")
            rejected_reason = ", ".join(reasons)

        editor_query_isolated = True
        if case.mode == MODE_EDITOR and case.explicit_query.strip():
            editor_query_isolated = final_query == _build_simple_search_query(
                mode=MODE_EDITOR,
                explicit_query=case.explicit_query,
                title="totally different title",
                body="irrelevant different body",
            )

        row = {
            "case_id": case.case_id,
            "input_mode": case.mode,
            "explicit_query": case.explicit_query,
            "title": case.title,
            "final_search_query": final_query,
            "provider": chosen_provider,
            "top3_candidates": [
                {"caption": c.caption, "tags": c.tags, "url": c.url} for c in candidates[:3]
            ],
            "chosen_image": result.media_ref,
            "final_outcome": outcome,
            "rejected_reason": rejected_reason,
            "editor_query_isolated": editor_query_isolated,
        }
        rows.append(row)
        if emit:
            print(json.dumps(row, ensure_ascii=False))

    summary = {"wrong_image": wrong, "no_image": no_image, "total": len(rows)}
    if emit:
        print(json.dumps({"summary": summary}, ensure_ascii=False))

    if enforce_gate:
        assert wrong <= 2, f"merge gate failed: wrong_image={wrong}/20"
        assert all(r["editor_query_isolated"] for r in rows if r["input_mode"] == MODE_EDITOR), (
            "merge gate failed: editor query affected by non-query fields"
        )

    return {"rows": rows, "summary": summary}


def main() -> None:
    asyncio.run(run_replay(emit=True, enforce_gate=True))


if __name__ == "__main__":
    main()
