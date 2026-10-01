"""The Jev questions for roadmap step 6, frozen before the first gold call
(docs/plan.md step 6; committed in bfcf76b, where they lived in this file).

Once the rule passed they moved verbatim into the runtime, coach/jev.py, so
the server sends exactly the request that was measured (the move was checked
by building the request for all 598 teacher rows both ways: identical). This
module re-exports them for the harness and the tests.
"""

from coach.jev import (GRADE_INSTRUCTIONS, GRADE_LEVELS, KP_CRITERIA,  # noqa: F401
                       KP_LABELS, build_questions, build_request, build_state,
                       parse_answers)
