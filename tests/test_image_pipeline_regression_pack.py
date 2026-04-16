from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visual_profile_layer import ProviderCandidate, build_visual_profile, profile_search_queries, score_candidate


DOMAIN_CASES = {
    "finance": [
        "Bank deposit rates for young families",
        "How to avoid hidden fees on debit cards",
        "Crypto risk checklist for beginners",
        "Mortgage pre-approval mistakes",
        "ETF portfolio rebalance strategy",
    ],
    "cars": [
        "When to replace brake pads",
        "Used sedan inspection before purchase",
        "Engine overheating in summer traffic",
        "Winter tire pressure routine",
        "Garage checklist before road trip",
    ],
    "scooter": [
        "Urban scooter suspension noise fix",
        "Best e-scooter tires for rain",
        "Commuter scooter battery care",
        "Night safety gear for scooter riders",
        "Kick scooter brakes quick tuning",
    ],
    "health": [
        "Annual blood test routine explained",
        "How to choose a family doctor",
        "Physical therapy after knee pain",
        "Hydration habits for office workers",
        "Sleep hygiene plan for shift workers",
    ],
    "local_news": [
        "City district opens new bus lane",
        "Community center schedule update",
        "Local library weekend workshop",
        "Municipal water service notice",
        "Neighborhood road repair timeline",
    ],
    "gardening": [
        "Seed starting calendar for spring",
        "Compost mix for tomato beds",
        "Soil pH basics for backyard gardens",
        "Balcony herbs watering routine",
        "Harvest timing for cucumbers",
    ],
    "electronics": [
        "Laptop battery health myths",
        "Smartphone camera sensor comparison",
        "Home router placement guide",
        "SSD upgrade compatibility check",
        "Noise-free PC cooling setup",
    ],
    "food": [
        "Quick pasta dinner with vegetables",
        "Healthy breakfast prep in 10 minutes",
        "How to bake crispy salmon",
        "Restaurant-style ramen at home",
        "Low-sugar dessert plating ideas",
    ],
    "education": [
        "Exam revision schedule for students",
        "Classroom engagement without gadgets",
        "Teacher feedback rubric examples",
        "Online course note-taking method",
        "Science project presentation tips",
    ],
    "beauty": [
        "Sensitive skin morning routine",
        "Salon hygiene checklist for clients",
        "How to layer skincare serums",
        "Hair mask weekly schedule",
        "Makeup brush cleaning protocol",
    ],
    "pets": [
        "Dog vaccination calendar basics",
        "Cat litter box odor control",
        "Puppy leash training milestones",
        "Pet grooming at home safely",
        "When to visit a vet urgently",
    ],
    "real_estate": [
        "Apartment viewing checklist",
        "First-time homebuyer mortgage prep",
        "Small studio staging techniques",
        "Rental contract red flags",
        "Neighborhood amenities audit",
    ],
}


def _candidate_set(domain: str, query: str, accept: bool) -> list[ProviderCandidate]:
    if accept:
        return [
            ProviderCandidate(
                url=f"https://images.pexels.com/{domain}/hero.jpg",
                provider="pexels",
                caption=f"{domain} editorial scene with realistic context",
                tags=[domain, "editorial", "realistic", "professional"],
                author="studio",
                source_query=query,
                width=1920,
                height=1280,
            ),
            ProviderCandidate(
                url=f"https://cdn.pixabay.com/{domain}/2.jpg",
                provider="pixabay",
                caption="generic office stock photo handshake",
                tags=["stock photo", "generic office"],
                author="stock",
                source_query=query,
                width=1600,
                height=900,
            ),
        ]
    return [
        ProviderCandidate(
            url=f"https://cdn.pixabay.com/bad/{domain}/1.jpg",
            provider="pixabay",
            caption="generic office stock photo handshake thumbs up",
            tags=["stock photo", "generic office", "handshake"],
            author="stock",
            source_query=query,
            width=1600,
            height=900,
        )
    ]


class TestImagePipelineRegressionPack(unittest.TestCase):
    def test_live_like_regression_pack_min_50(self):
        report: list[dict[str, object]] = []
        idx = 0
        for domain, titles in DOMAIN_CASES.items():
            for title in titles:
                idx += 1
                profile = build_visual_profile(
                    title=title,
                    channel_topic=domain,
                    onboarding_summary=f"rubric for {domain} practical channel",
                    post_intent="educational guidance",
                    body="concise body details and constraints",
                )
                primary_q, backup_q = profile_search_queries(profile)
                # Force roughly every 5th case to be no_image by giving only generic stock.
                accept = (idx % 5) != 0
                query_for_candidates = primary_q if accept else "generic office people handshake"
                top_candidates = _candidate_set(domain, query_for_candidates, accept=accept)
                threshold = 1.5 if accept else 3.0
                scored = [score_candidate(candidate=c, profile=profile, min_score=threshold) for c in top_candidates]
                best = max(scored, key=lambda s: s.score)
                final_decision = "accepted" if best.decision == "accepted" else "no_image"
                rejected_reason = "" if final_decision == "accepted" else best.reason

                report.append(
                    {
                        "title": title,
                        "visual_profile": {
                            "domain_family": profile.domain_family,
                            "primary_subject": profile.primary_subject,
                            "secondary_subjects": profile.secondary_subjects,
                            "scene_type": profile.scene_type,
                            "visual_must_have": profile.visual_must_have,
                            "visual_must_not_have": profile.visual_must_not_have,
                            "search_terms_primary": profile.search_terms_primary,
                            "search_terms_backup": profile.search_terms_backup,
                        },
                        "final_search_queries": [primary_q, backup_q],
                        "top_candidates": [
                            {
                                "caption": c.caption,
                                "tags": c.tags,
                                "provider": c.provider,
                                "url": c.url,
                                "source_query": c.source_query,
                            }
                            for c in top_candidates
                        ],
                        "final_score": best.score,
                        "outcome": final_decision,
                        "rejected_reason": rejected_reason,
                    }
                )

        self.assertGreaterEqual(len(report), 50)
        for item in report:
            self.assertIn("visual_profile", item)
            self.assertIn("final_search_queries", item)
            self.assertIn("top_candidates", item)
            self.assertIn("final_score", item)
            self.assertIn(item["outcome"], {"accepted", "no_image"})

        accepted = sum(1 for r in report if r["outcome"] == "accepted")
        no_image = sum(1 for r in report if r["outcome"] == "no_image")
        self.assertGreater(accepted, 0)
        self.assertGreater(no_image, 0)


if __name__ == "__main__":
    unittest.main()
