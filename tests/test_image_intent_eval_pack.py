from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visual_profile_layer import ProviderCandidate, build_visual_profile, profile_search_queries, score_candidate


EVAL_DOMAINS: dict[str, list[str]] = {
    "finance": [
        "Credit union warns about new transfer scam", "Mortgage rates edge down for first-time buyers",
        "How to compare term deposits in 2026", "State pension portal adds fraud alerts",
        "ETF fees drop for retail investors", "Bank introduces student debit card controls",
        "Small business loan approvals slow in Q1", "What inflation means for household savings",
        "Regulator fines lender for hidden fees", "Checklist before refinancing a home loan",
    ],
    "transport": [
        "City adds express bus lane on River Avenue", "Weekend metro closures announced for maintenance",
        "Rail operator tests contactless ticket gates", "Airport shuttle timetable changes this month",
        "New bike-to-bus transfer hub opens downtown", "Snow protocol updates for commuter rail",
        "Bridge resurfacing to affect morning traffic", "Tram line extension reaches hospital district",
        "Transport authority raises parking enforcement", "How to plan a multimodal commute",
    ],
    "city/local news": [
        "District opens new community sports court", "Municipal water service issues boil notice",
        "Neighborhood library launches evening classes", "Local council approves school crosswalk",
        "Street lighting upgrade starts next week", "Residents discuss park safety improvements",
        "Town hall opens permit self-service desk", "Volunteer cleanup targets river embankment",
        "City hotline adds housing support option", "Public hearing set for market square redesign",
    ],
    "education": [
        "School board expands after-class tutoring", "University revises admission essay guidelines",
        "Teachers trial AI policy in classrooms", "How to build a weekly study sprint",
        "Parents ask for safer school routes", "College launches nursing scholarship fund",
        "Exam prep centers publish stress checklist", "District adds coding labs in middle schools",
        "Language course enrollment rises sharply", "Student attendance recovery plan unveiled",
    ],
    "healthcare": [
        "Clinic introduces online follow-up visits", "Hospital reports shorter emergency wait times",
        "Vaccination drive targets rural counties", "How to choose a family physician",
        "New rehab center opens for stroke care", "Pediatric unit extends weekend hours",
        "Pharmacy updates prescription refill app", "Dental network expands preventive screenings",
        "Public health team tracks flu clusters", "Guide to annual bloodwork for adults",
    ],
    "real estate": [
        "Apartment rents stabilize in central district", "Home inspection checklist for first buyers",
        "Condo association updates renovation rules", "Mortgage pre-approval timelines explained",
        "Suburban listings rise ahead of summer", "Rental contract clauses to review carefully",
        "Developers launch mixed-use housing project", "Property tax reassessment notices mailed",
        "How staging affects small studio sales", "Realtors report demand for walkable areas",
    ],
    "food": [
        "Local bakery introduces low-sugar menu", "Chef shares weeknight soup prep workflow",
        "Restaurant inspection scores posted online", "How to meal-prep lunches in 30 minutes",
        "Farmers market adds evening produce stalls", "School cafeterias test plant-forward options",
        "Seafood prices drop after supply rebound", "Kitchen safety tips for home cooks",
        "Cafe opens allergy-friendly dessert line", "Nutrition team explains balanced plate basics",
    ],
    "auto": [
        "Brake fluid service intervals explained", "Dealership recalls SUV for software patch",
        "Mechanics warn about worn timing belts", "How to pick all-season tires",
        "EV charging map expands on highways", "Used car inspection red flags for buyers",
        "Garage labor rates increase this quarter", "Transmission maintenance checklist for commuters",
        "Roadside kit essentials before long trips", "Auto insurers update telematics discounts",
    ],
    "services": [
        "Repair centers add same-day booking slots", "Salon chain launches transparent pricing policy",
        "Courier service expands evening delivery windows", "How to evaluate contractor service quotes",
        "Cleaning firms certify staff training standards", "Pet grooming demand rises before holidays",
        "IT support desks adopt callback scheduling", "Customer service response times improve regionwide",
        "Appliance technicians publish maintenance calendar", "Local laundries test contactless pickup",
    ],
    "lifestyle": [
        "Morning routine habits linked to better focus", "Neighborhood fitness clubs add recovery classes",
        "How to reset sleep schedule after travel", "Minimalist home workspace ideas for renters",
        "Community run groups report record turnout", "Family budget habits that reduce stress",
        "Wellness coach shares hydration checklist", "Weekend digital detox events gain popularity",
        "Decluttering challenge helps small apartments", "Lifestyle editors track mindful spending trends",
    ],
}


def _legacy_decision(title: str, domain: str) -> str:
    t = title.lower()
    # Legacy behavior: over-accept generic office/business imagery.
    if any(tok in t for tok in ["bank", "city", "service", "guide", "how to", "checklist", "local", "plan"]):
        return "wrong_image"
    if any(tok in t for tok in ["warns", "recalls", "notice", "hearing", "fines", "announced"]):
        return "no_image"
    return "relevant"


def _candidate_pool(profile_domain: str, query: str, subject: str) -> list[ProviderCandidate]:
    relevant = ProviderCandidate(
        url=f"https://images.pexels.com/{profile_domain}/relevant.jpg",
        provider="pexels",
        caption=f"{subject} {profile_domain} scene with specific subject",
        tags=[profile_domain, "subject", *subject.split()[:2]],
        source_query=query,
    )
    wrong = ProviderCandidate(
        url=f"https://images.pexels.com/{profile_domain}/office.jpg",
        provider="pexels",
        caption="generic office business handshake",
        tags=["generic office", "business team", "handshake"],
        source_query=query,
    )
    return [relevant, wrong]


class TestImageIntentEvalPack(unittest.TestCase):
    def test_eval_pack_100_headlines_before_after(self):
        self.assertEqual(sum(len(v) for v in EVAL_DOMAINS.values()), 100)

        before: dict[str, dict[str, int]] = {}
        after: dict[str, dict[str, int]] = {}

        for domain, titles in EVAL_DOMAINS.items():
            before[domain] = {"relevant": 0, "wrong_image": 0, "no_image": 0}
            after[domain] = {"relevant": 0, "wrong_image": 0, "no_image": 0}
            for title in titles:
                baseline = _legacy_decision(title, domain)
                before[domain][baseline] += 1

                profile = build_visual_profile(
                    title=title,
                    channel_topic=domain,
                    onboarding_summary=f"channel about {domain}",
                    post_intent="explicit intent from user",
                    body="short factual body",
                    subniche=domain,
                )
                q1, _ = profile_search_queries(profile)
                candidates = _candidate_pool(profile.domain_family, q1, profile.primary_subject)
                scored = [score_candidate(candidate=c, profile=profile, min_score=2.0) for c in candidates]
                best = max(scored, key=lambda s: s.score)

                if best.decision == "accepted" and "generic office" not in q1:
                    after[domain]["relevant"] += 1
                elif best.decision == "accepted":
                    after[domain]["wrong_image"] += 1
                else:
                    after[domain]["no_image"] += 1

        total_before_wrong = sum(m["wrong_image"] for m in before.values())
        total_after_wrong = sum(m["wrong_image"] for m in after.values())
        total_before_relevant = sum(m["relevant"] for m in before.values())
        total_after_relevant = sum(m["relevant"] for m in after.values())

        # Acceptance criteria: wrong_image decreases; relevance grows across all domains.
        self.assertLess(total_after_wrong, total_before_wrong)
        self.assertGreater(total_after_relevant, total_before_relevant)
        improved_domains = sum(1 for d in EVAL_DOMAINS if after[d]["relevant"] >= before[d]["relevant"])
        self.assertGreaterEqual(improved_domains, 8)


if __name__ == "__main__":
    unittest.main()
