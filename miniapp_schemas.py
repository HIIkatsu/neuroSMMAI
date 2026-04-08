from __future__ import annotations

from pydantic import BaseModel, Field


class ChannelCreate(BaseModel):
    title: str = ""
    channel_target: str
    topic: str = ""
    make_active: bool = True


class ChannelActivate(BaseModel):
    profile_id: int


class DraftCreate(BaseModel):
    text: str = ""
    prompt: str = ""
    topic: str = ""
    channel_target: str = ""
    media_type: str = "none"
    media_ref: str = ""
    media_meta_json: str = ""
    buttons_json: str = "[]"
    pin_post: int = 0
    comments_enabled: int = 1
    ad_mark: int = 0
    first_reaction: str = ""
    reply_to_message_id: int = 0
    send_silent: int = 0


class DraftUpdate(BaseModel):
    text: str | None = None
    prompt: str | None = None
    topic: str | None = None
    channel_target: str | None = None
    media_type: str | None = None
    media_ref: str | None = None
    media_meta_json: str | None = None
    buttons_json: str | None = None
    pin_post: int | None = None
    comments_enabled: int | None = None
    ad_mark: int | None = None
    first_reaction: str | None = None
    reply_to_message_id: int | None = None
    status: str | None = None
    send_silent: int | None = None


class DraftPublish(BaseModel):
    draft_id: int
    mirror_targets: list[str] = []  # Additional channels for linked publishing


class DraftGenerate(BaseModel):
    prompt: str


class AIGenerateText(BaseModel):
    prompt: str
    topic: str = ""


class AIGeneratePost(BaseModel):
    prompt: str = ""
    topic: str = ""
    current_text: str = ""
    channel_target: str = ""
    draft_id: int | None = None
    media_type: str = "none"
    media_ref: str = ""
    media_meta_json: str = ""
    force_image: bool = False
    buttons_json: str = "[]"
    pin_post: int = 0
    comments_enabled: int = 1
    ad_mark: int = 0


class AIAddHashtags(BaseModel):
    text: str = ""
    topic: str = ""
    prompt: str = ""


class AIRewrite(BaseModel):
    text: str = ""
    prompt: str = ""
    topic: str = ""
    mode: str = "improve"


class AIAssets(BaseModel):
    text: str = ""
    prompt: str = ""
    topic: str = ""


class PlanGenerate(BaseModel):
    start_date: str | None = None
    days: int = Field(default=30, ge=1, le=90)
    posts_per_day: int = Field(default=1, ge=1, le=4)
    topic: str = ""
    post_time: str = "12:00"
    clear_existing: bool = True


class PlanCreate(BaseModel):
    dt: str
    topic: str = ""
    prompt: str = ""


class PlanUpdate(BaseModel):
    dt: str | None = None
    topic: str | None = None
    prompt: str | None = None


class ScheduleCreate(BaseModel):
    time_hhmm: str
    days: str = "*"


class SettingsUpdate(BaseModel):
    posts_enabled: str | None = None
    posting_mode: str | None = None
    news_enabled: str | None = None
    news_interval_hours: str | None = None
    news_sources: str | None = None
    auto_hashtags: str | None = None
    topic: str | None = None
    channel_style: str | None = None
    content_rubrics: str | None = None
    rubrics_schedule: str | None = None
    post_scenarios: str | None = None
    onboarding_completed: str | None = None
    channel_audience: str | None = None
    channel_style_preset: str | None = None
    channel_mode: str | None = None
    channel_formats: str | None = None
    channel_frequency: str | None = None
    content_constraints: str | None = None
    content_exclusions: str | None = None
    news_strict_mode: str | None = None
    # Author role fields
    author_role_type: str | None = None
    author_role_description: str | None = None
    author_activities: str | None = None
    author_forbidden_claims: str | None = None


class AIAnalyticsRequest(BaseModel):
    focus: str = "общий разбор"
