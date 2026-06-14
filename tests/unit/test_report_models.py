from sqlalchemy import inspect

from app.db.models import Report, ReportJob


def test_report_rows_reference_transcription_workflow() -> None:
    expected = {"transcription_workflows.transcription_id"}

    assert {str(key.column) for key in ReportJob.source_transcription_id.property.columns[0].foreign_keys} == expected
    assert {str(key.column) for key in Report.source_transcription_id.property.columns[0].foreign_keys} == expected


def test_report_job_has_queue_claim_index() -> None:
    indexes = {
        index.name: tuple(column.name for column in index.columns) for index in inspect(ReportJob).local_table.indexes
    }

    assert indexes["ix_report_jobs_status_created_at"] == ("status", "created_at")
