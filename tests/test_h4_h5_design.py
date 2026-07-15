import math
import unittest

from hypotheses.h0_carrier_norming import _collapse_mirrored_carriers
from hypotheses.h4_dissociation import conditional_transition_summary
from hypotheses.h5_garden_path import audit_h5_design, build_incremental_prefixes


class H0CarrierTests(unittest.TestCase):
    def test_collapses_mirrored_sense_rows_to_independent_carrier(self):
        records = [
            {
                "word": "bank",
                "carrier": "The bank was quiet.",
                "sense": 0,
                "M_l_carrier": 2.0,
                "layer": 3,
            },
            {
                "word": "bank",
                "carrier": "The bank was quiet.",
                "sense": 1,
                "M_l_carrier": -2.0,
                "layer": 3,
            },
        ]

        rows = _collapse_mirrored_carriers(
            records,
            word_alone_raw=1.0,
            scale=4.0,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["signed_M_l_carrier_raw"], 2.0)
        self.assertEqual(rows[0]["signed_M_l_carrier_norm"], 0.5)
        self.assertEqual(rows[0]["signed_carrier_shift_raw"], 1.0)
        self.assertEqual(rows[0]["prior_direction"], "sense_0")


class H4TransitionTests(unittest.TestCase):
    def test_reports_full_conditional_transition_table(self):
        summary = conditional_transition_summary(
            [False, False, True, True],
            [False, True, False, True],
        )

        self.assertEqual(summary["n_target_inadequate_final_inadequate"], 1)
        self.assertEqual(summary["n_target_inadequate_final_adequate"], 1)
        self.assertEqual(summary["n_target_adequate_final_inadequate"], 1)
        self.assertEqual(summary["n_target_adequate_final_adequate"], 1)
        self.assertEqual(summary["p_final_adequate_given_target_inadequate"], 0.5)
        self.assertEqual(summary["p_final_inadequate_given_target_adequate"], 0.5)

    def test_empty_conditioning_set_is_not_reported_as_zero(self):
        summary = conditional_transition_summary([True, True], [True, False])
        self.assertEqual(summary["n_target_inadequate"], 0)
        self.assertTrue(math.isnan(summary["p_final_adequate_given_target_inadequate"]))


class H5DesignTests(unittest.TestCase):
    def test_prefixes_hold_readout_identity_and_role_constant(self):
        item = {
            "id": "bank_test",
            "sentence": "She discussed a loan at the bank before reaching the river.",
            "primed_sense": 1,
            "correct_sense": 0,
            "resolution_word": "river",
        }
        prefixes = build_incremental_prefixes(item, "bank", sentinel="probe")

        self.assertTrue(prefixes["prime"].endswith("\n\nprobe"))
        self.assertTrue(prefixes["homonym"].endswith("\n\nprobe"))
        self.assertTrue(prefixes["resolution"].endswith("\n\nprobe"))
        self.assertNotIn("bank", prefixes["prime"].split("\n\n")[0])
        self.assertIn("bank", prefixes["homonym"].split("\n\n")[0])
        self.assertIn("river", prefixes["resolution"].split("\n\n")[0])

    def test_audit_requires_directions_and_controls_but_not_human_norms(self):
        base = {
            "sentence": "A loan led her to the bank beside the river.",
            "resolution_word": "river",
            "matched_control_sentence": "A hiker rested on the bank beside the river.",
        }
        data = {
            "bank": [
                {**base, "id": "a", "primed_sense": 1, "correct_sense": 0},
                {
                    **base,
                    "id": "b",
                    "sentence": "A fisherman saw the bank approve her loan.",
                    "resolution_word": "loan",
                    "matched_control_sentence": "A client saw the bank approve her loan.",
                    "primed_sense": 0,
                    "correct_sense": 1,
                },
            ]
        }
        rows, ready = audit_h5_design(data, words=["bank"])
        self.assertTrue(ready)
        self.assertTrue(rows[0]["model_internal_ready"])
        self.assertFalse(rows[0]["human_norms_complete"])
        self.assertEqual(
            rows[0]["human_validation_status"], "not_collected_not_required"
        )

        rows, ready = audit_h5_design(data, words=["bank", "light"])
        self.assertTrue(ready)
        light_row = next(row for row in rows if row["word"] == "light")
        self.assertFalse(light_row["eligible_for_h5"])
        self.assertIn("parts of speech", light_row["exclusion_reason"])

        del data["bank"][0]["matched_control_sentence"]
        rows, ready = audit_h5_design(data, words=["bank"])
        self.assertFalse(ready)
        self.assertFalse(rows[0]["matched_controls_complete"])


if __name__ == "__main__":
    unittest.main()
