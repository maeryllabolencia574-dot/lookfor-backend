import unittest
from datetime import date
from types import SimpleNamespace

from student_routes import (
    _student_item_matches_lifecycle_view,
    delete_my_report,
    recover_my_report,
    uses_student_personal_lifecycle,
)


class _Query:
    def __init__(self, result):
        self.result = result

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self.result


class _Session:
    def __init__(self, *results):
        self.results = list(results)
        self.committed = False

    def query(self, *args, **kwargs):
        result = self.results.pop(0) if self.results else None
        return _Query(result)

    def commit(self):
        self.committed = True


def _report(*, record_type="found", surrendered=False, matched=False):
    return SimpleNamespace(
        id=41,
        status=record_type,
        user_id=7,
        report_owner_user_id=None,
        is_surrendered=surrendered,
        is_matched=matched,
        archived=False,
        deleted=False,
        student_archived=False,
        student_deleted=False,
        student_hidden=False,
        item_name="Test report",
        category="Other",
        description=None,
        image_path=None,
        image_embedding=None,
        brand=None,
        color=None,
        location="Library",
        date=date(2026, 1, 1),
        time_found=None,
        possible_matches=None,
        category_id=None,
        created_at=None,
        user=None,
        owner=None,
    )


class StudentDeleteLifecycleTests(unittest.TestCase):
    def test_for_surrender_uses_shared_admin_lifecycle(self):
        report = _report(surrendered=False)
        self.assertFalse(
            uses_student_personal_lifecycle(
                _Session(), report, "found", is_claimed=False
            )
        )

    def test_surrendered_and_matched_items_use_personal_lifecycle(self):
        self.assertTrue(
            uses_student_personal_lifecycle(
                _Session(), _report(surrendered=True), "found", is_claimed=False
            )
        )
        self.assertTrue(
            uses_student_personal_lifecycle(
                _Session(), _report(matched=True), "found", is_claimed=False
            )
        )

    def test_unmatched_lost_uses_shared_but_matched_lost_is_personal(self):
        self.assertFalse(
            uses_student_personal_lifecycle(
                _Session(), _report(record_type="lost"), "lost", is_claimed=False
            )
        )
        self.assertTrue(
            uses_student_personal_lifecycle(
                _Session(), _report(record_type="lost", matched=True), "lost", is_claimed=False
            )
        )

    def test_deleting_for_surrender_hides_it_from_both_active_lists(self):
        report = _report(surrendered=False)
        db = _Session(report, None)

        delete_my_report("found", report.id, db, SimpleNamespace(id=7))

        self.assertTrue(report.archived)
        self.assertTrue(report.deleted)
        self.assertFalse(report.student_deleted)
        self.assertTrue(db.committed)

    def test_recovering_for_surrender_restores_shared_visibility(self):
        report = _report(surrendered=False)
        report.archived = True
        report.deleted = True
        db = _Session(report, None)

        recover_my_report("found", report.id, db, SimpleNamespace(id=7))

        self.assertFalse(report.archived)
        self.assertFalse(report.deleted)
        self.assertTrue(db.committed)

    def test_shared_deleted_item_is_in_deleted_view_but_not_active_view(self):
        report = _report(surrendered=False)
        report.archived = True
        report.deleted = True

        self.assertTrue(
            _student_item_matches_lifecycle_view(
                report, True, True, personal_lifecycle=False
            )
        )
        self.assertFalse(
            _student_item_matches_lifecycle_view(
                report, False, False, personal_lifecycle=False
            )
        )

    def test_personal_deleted_item_uses_student_flags_and_preserves_admin_state(self):
        report = _report(surrendered=True)
        report.student_archived = True
        report.student_deleted = True

        self.assertTrue(
            _student_item_matches_lifecycle_view(
                report, True, True, personal_lifecycle=True
            )
        )
        self.assertFalse(
            _student_item_matches_lifecycle_view(
                report, False, False, personal_lifecycle=True
            )
        )

    def test_deleting_surrendered_item_keeps_admin_record_active(self):
        report = _report(surrendered=True)
        db = _Session(report, None)

        delete_my_report("found", report.id, db, SimpleNamespace(id=7))

        self.assertFalse(report.archived)
        self.assertFalse(report.deleted)
        self.assertTrue(report.student_archived)
        self.assertTrue(report.student_deleted)


if __name__ == "__main__":
    unittest.main()
