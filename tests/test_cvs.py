from examples.cvs import cvs


class TestCVs:
    def test_cvs_count(self):
        assert len(cvs) == 1200

    def test_cv_has_metadata(self):
        for cv in cvs:
            assert "metadata" in cv
            meta = cv["metadata"]
            for key in ("name", "university", "school", "school_location", "company", "a_levels"):
                assert key in meta

    def test_cv_has_template_text(self):
        for cv in cvs:
            assert "cv" in cv
            assert isinstance(cv["cv"], str)
            assert len(cv["cv"]) > 0
