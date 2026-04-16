from miniapp_analytics_service import build_channel_analytics


def test_score_marked_insufficient_when_almost_no_data():
    payload = build_channel_analytics(
        stats={"total_posts": 0, "posted_last_7d": 0},
        history={"recent_posts": [], "recent_drafts": [], "recent_plan": [], "recent_generations": []},
        analytics_snapshot={},
        settings={},
        active_channel=None,
        drafts=[],
        plan_items=[],
        media_items=[],
        schedules=[],
    )

    assert payload["score_status"] == "insufficient"
    assert payload["summary"]["score_status"] == "insufficient"
    assert payload["summary"]["available_factors"] <= 1


def test_score_breakdown_contains_required_factors():
    payload = build_channel_analytics(
        stats={"total_posts": 5, "posted_last_7d": 3},
        history={"recent_posts": ["a", "b", "c"], "recent_drafts": ["d"], "recent_plan": [], "recent_generations": []},
        analytics_snapshot={"top_topics": ["A", "B", "C"], "views_known": True, "avg_views": 100},
        settings={"topic": "AI", "channel_audience": "founders", "channel_style": "short", "channel_formats": '["разбор"]'},
        active_channel={"id": 1},
        drafts=[{"status": "draft"}],
        plan_items=[{"posted": 0}],
        media_items=[],
        schedules=[{"enabled": 1}],
    )

    keys = {item["key"] for item in payload.get("score_factors", [])}
    assert {"regularity", "volume", "activity", "topics", "plan"}.issubset(keys)
    assert payload["score_status"] in {"preliminary", "stable"}
