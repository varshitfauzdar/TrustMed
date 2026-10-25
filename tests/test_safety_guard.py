import unittest

from modules.safety_guard import get_system_prompt


class TestSafetyGuard(unittest.TestCase):

    def test_system_prompt_is_not_empty(self):
        prompt = get_system_prompt()

        self.assertIsInstance(prompt, str)
        self.assertTrue(prompt.strip())

    def test_system_prompt_mentions_medical_safety(self):
        prompt = get_system_prompt().lower()

        self.assertTrue(
            any(
                term in prompt
                for term in [
                    "medical",
                    "healthcare",
                    "health",
                    "doctor",
                    "clinical",
                ]
            ),
            "System prompt should include healthcare safety guidance.",
        )


if __name__ == "__main__":
    unittest.main()
