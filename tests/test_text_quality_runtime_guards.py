import unittest

from text_validator import validate_generated_text


class TestTextQualityRuntimeGuards(unittest.TestCase):
    def test_duplicate_paragraph_reject(self):
        text = (
            "Как выбрать подрядчика для ремонта без переплат.\n\n"
            "Сначала зафиксируйте объём работ и список материалов, чтобы не спорить по ходу проекта.\n\n"
            "Сначала зафиксируйте объём работ и список материалов, чтобы не спорить по ходу проекта."
        )
        result = validate_generated_text(text, generation_mode="manual")
        self.assertTrue(result.should_reject)
        self.assertIn("duplicate_paragraph", result.text_quality_hits)

    def test_duplicate_final_question_reject(self):
        text = (
            "Проверьте договор и этапы оплаты до старта.\n"
            "Вы проверили этапы оплаты?\n"
            "Вы проверили этапы оплаты?"
        )
        result = validate_generated_text(text, generation_mode="manual")
        self.assertTrue(result.should_reject)
        self.assertIn("duplicate_final_question", result.text_quality_hits)

    def test_fabricated_numeric_claim_reject(self):
        text = "По данным аналитиков, 64% клиентов теряют деньги из-за этой ошибки."
        result = validate_generated_text(text, generation_mode="manual")
        self.assertTrue(result.should_reject)
        self.assertGreater(len(result.fake_numeric_claims), 0)

    def test_cheap_clickbait_penalty(self):
        text = "ШОК: на что смотреть в договоре перед покупкой квартиры."
        result = validate_generated_text(text, generation_mode="autopost", reject_threshold=6)
        self.assertFalse(result.should_reject)
        self.assertGreater(len(result.clickbait_hits), 0)
        self.assertGreater(result.total_risk_score, 0)

    def test_honest_clean_text_passes(self):
        text = (
            "Перед покупкой квартиры проверьте выписку ЕГРН, историю перехода прав и ограничения.\n\n"
            "Попросите у продавца документы заранее и сверяйте адрес, площадь и кадастровый номер."
        )
        result = validate_generated_text(text, generation_mode="manual")
        self.assertFalse(result.should_reject)
        self.assertEqual(result.total_risk_score, 0)


if __name__ == "__main__":
    unittest.main()
