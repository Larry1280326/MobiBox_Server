"""API routes for query endpoints."""

from fastapi import APIRouter, HTTPException

from src.query.schemas import (
    SummaryLogRequest,
    SummaryLogResponse,
    SummaryLogItem,
    InterventionRequest,
    InterventionResponse,
    InterventionItem,
    InterventionFeedbackRequest,
    InterventionFeedbackResponse,
    SummaryLogFeedbackRequest,
    SummaryLogFeedbackResponse,
    AtomicActivitiesRequest,
    AtomicActivitiesResponse,
    AtomicActivitiesData,
)
from src.query.service import (
    get_summary_logs,
    get_interventions,
    format_summary_log,
    format_intervention,
    submit_intervention_feedback,
    submit_summary_log_feedback,
    get_atomic_activities,
)

router = APIRouter(tags=["query"])


@router.post("/get_summary_log", response_model=SummaryLogResponse)
async def fetch_summary_log(request: SummaryLogRequest):
    """
    Fetch the most recent summary log for a user.

    Given a user ID and log type, returns the most recent summary log content,
    window timestamps, and generation timestamp.

    Supports polling: provide last_log_id to detect if there's a new log.
    - If last_log_id is provided and matches the latest log ID, returns has_new_log=False with data=None
    - If there's a newer log, returns has_new_log=True with the new log data
    - If last_log_id is not provided, returns the latest log (has_new_log=True)
    """
    try:
        record, has_new_log = await get_summary_logs(
            user=request.user,
            log_type=request.log_type,
            last_log_id=request.last_log_id,
        )

        if record is None:
            return SummaryLogResponse(status="success", data=None, has_new_log=has_new_log)

        formatted = format_summary_log(record)
        return SummaryLogResponse(
            status="success",
            data=SummaryLogItem(**formatted),
            has_new_log=has_new_log,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get_intervention", response_model=InterventionResponse)
async def fetch_intervention(request: InterventionRequest):
    """
    Fetch the most recent intervention for a user.

    Given a user ID, returns the most recent intervention content,
    window timestamps, and generation timestamp.
    """
    try:
        record = await get_interventions(user=request.user)

        if record is None:
            return InterventionResponse(status="success", data=None)

        formatted = format_intervention(record)
        return InterventionResponse(
            status="success",
            data=InterventionItem(**formatted),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/send_intervention_feedback", response_model=InterventionFeedbackResponse)
async def send_intervention_feedback(request: InterventionFeedbackRequest):
    """
    Submit feedback for an intervention.

    Receives user feedback for a specific intervention and stores it in the database.
    """
    try:
        await submit_intervention_feedback(
            user=request.user,
            intervention_id=request.intervention_id,
            feedback=request.feedback,
            mc1=request.mc1,
            mc2=request.mc2,
            mc3=request.mc3,
            mc4=request.mc4,
            mc5=request.mc5,
            mc6=request.mc6,
        )
        return InterventionFeedbackResponse()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/send_log_feedback", response_model=SummaryLogFeedbackResponse)
async def send_log_feedback(request: SummaryLogFeedbackRequest):
    """
    Submit feedback for a summary log.

    Receives user feedback for a specific summary log and stores it in the database.
    """
    try:
        await submit_summary_log_feedback(
            user=request.user,
            summary_logs_id=request.summary_logs_id,
            feedback=request.feedback,
        )
        return SummaryLogFeedbackResponse()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get_compressed_atomic_activities", response_model=AtomicActivitiesResponse)
async def fetch_atomic_activities(request: AtomicActivitiesRequest):
    """
    Fetch compressed atomic activities for a user.

    Returns atomic activity labels grouped by type:
    - sport: HAR activity labels
    - appCategory: App usage categories
    - location: Location context labels
    - movement: Movement pattern labels
    - stepCategory: Step activity labels
    - phoneCategory: Phone usage labels

    Args:
        request: Contains user identifier and duration (seconds since last fetch)

    Returns:
        AtomicActivitiesResponse with grouped activity data
    """
    try:
        data = await get_atomic_activities(
            user=request.user,
            duration=request.duration,
        )

        return AtomicActivitiesResponse(
            status="success",
            data=AtomicActivitiesData(**data),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))