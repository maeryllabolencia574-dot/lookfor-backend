from fastapi import APIRouter, Request, Response, UploadFile, File, Form, Depends
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
import models
from database import get_db
from security import get_current_user
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
import os
import shutil
import io
from datetime import datetime
import uuid
from clip_test import get_clip_components
from PIL import Image
import numpy as np
import json
from utils import (
    public_file_url,
    save_file,
    resolve_category_name,
    validate_upload_file_size,
    format_user_display_name,
    format_item_code,
    item_display_id,
    item_display_code,
)
from clip_test import get_text_embedding, get_image_embedding, get_multi_image_embedding
from models import SettingsUpdate
from account_email import queue_item_event_email


router = APIRouter(prefix="/student", tags=["Student"])
templates = Jinja2Templates(directory="templates")
STUDENT_ACCESS_PERMISSION = "Student-Portal-Access"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
STATIC_PROFILE_PICS_DIR = os.path.join(STATIC_DIR, "profile_pics")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_PROFILE_PICS_DIR, exist_ok=True)


def parse_permissions(raw_permissions) -> list[str]:
    try:
        if isinstance(raw_permissions, str):
            return json.loads(raw_permissions)
        return raw_permissions or []
    except Exception:
        return []


def get_active_student_user(
    current_user: models.User = Depends(get_current_user)
):
    if current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admins cannot access student actions")

    permissions = parse_permissions(current_user.permissions)
    if STUDENT_ACCESS_PERMISSION not in permissions:
        raise HTTPException(
            status_code=403,
            detail="Your student account is still deactivated. Please wait for admin activation."
        )

    return current_user


def create_student_notification(
    db: Session,
    user_id: int,
    message: str,
    notif_type: str = "student_match",
    target_url: str | None = None
):
    if target_url is None:
        if notif_type == "chat":
            target_url = "/student/Messages"
        elif notif_type in {"student_match", "student_update"}:
            target_url = "/student/Lost-report"

    notif = models.Notification(
        message=message,
        type=notif_type,
        related_id=user_id,
        target_url=target_url,
        is_read=False,
        created_at=datetime.utcnow()
    )
    db.add(notif)
    return notif


def queue_lost_item_match_email(
    recipient: models.User | None,
    lost_item: models.Item,
    *,
    subject: str = "A possible match was found for your lost item",
    message_text: str | None = None,
) -> bool:
    if not recipient or not getattr(recipient, "email", None):
        return False

    target_url = f"/student/Lost-report?item_id={lost_item.id}&show_match=1"
    return queue_item_event_email(
        recipient.email,
        format_user_display_name(recipient),
        subject=subject,
        message_text=(
            message_text
            or f"A found {lost_item.category} may match your lost-item report."
        ),
        action_url=target_url,
    )


def normalize_saved_possible_matches(raw_possible_matches: str | None) -> str | None:
    if not raw_possible_matches:
        return None

    try:
        parsed_matches = json.loads(raw_possible_matches)
    except Exception:
        return None

    if not isinstance(parsed_matches, list):
        return None

    cleaned_matches = []
    for match in parsed_matches[:3]:
        if not isinstance(match, dict):
            continue
        cleaned_matches.append({
            "id": match.get("id"),
            "score": match.get("score"),
            "item_name": match.get("item_name"),
            "category": match.get("category"),
            "location": match.get("location"),
            "image_path": public_file_url(match.get("image_path")),
            "brand": match.get("brand"),
            "color": match.get("color"),
            "description": match.get("description"),
            "source": match.get("source", "found"),
            "cross_category": bool(match.get("cross_category")),
            "warning": match.get("warning"),
        })

    return json.dumps(cleaned_matches) if cleaned_matches else None


def serialize_pending_found_match(pending_item: models.PendingItem, score: float | None = None) -> dict:
    return {
        "id": pending_item.id,
        "score": round(float(score or 0), 4),
        "item_name": pending_item.item_name,
        "category": pending_item.category,
        "location": pending_item.location,
        "image_path": public_file_url(pending_item.image_path),
        "brand": pending_item.brand,
        "color": pending_item.color,
        "description": pending_item.description,
        "source": "pending_found",
    }


def serialize_found_item_match(found_item: models.Item, score: float | None = None) -> dict:
    return {
        "id": found_item.id,
        "score": round(float(score or 0), 4),
        "item_name": found_item.item_name,
        "category": found_item.category,
        "location": found_item.location,
        "image_path": public_file_url(found_item.image_path),
        "brand": found_item.brand,
        "color": found_item.color,
        "description": found_item.description,
        "source": "found",
    }


def prepend_lost_possible_match(lost_item: models.Item, match_payload: dict) -> int:
    existing_matches = []
    if lost_item.possible_matches:
        try:
            parsed = json.loads(lost_item.possible_matches)
            existing_matches = parsed if isinstance(parsed, list) else []
        except Exception:
            existing_matches = []

    match_source = match_payload.get("source")
    match_id = match_payload.get("id")
    deduped_matches = [
        match for match in existing_matches
        if not (
            isinstance(match, dict)
            and match.get("id") == match_id
            and match.get("source", "found") == match_source
        )
    ]
    updated_matches = [match_payload, *deduped_matches][:3]
    lost_item.possible_matches = json.dumps(updated_matches)
    return len(updated_matches)


def ensure_student_claim_for_pair(
    db: Session,
    lost_item: models.Item,
    found_item: models.Item,
    claimant_id: int,
    similarity_score: str = ""
) -> models.Claim:
    existing_claim = db.query(models.Claim).filter(
        models.Claim.lost_item_id == lost_item.id,
        models.Claim.found_item_id == found_item.id,
        models.Claim.status.in_(models.ACTIVE_CLAIM_STATUSES)
    ).first()
    if existing_claim:
        return existing_claim

    new_claim = models.Claim(
        lost_item_id=lost_item.id,
        found_item_id=found_item.id,
        claimant_id=claimant_id,
        status="pending",
        similarity_score=similarity_score
    )
    db.add(new_claim)
    db.flush()
    return new_claim


def serialize_student_item(
    item: models.Item,
    owner: models.User | None,
    *,
    is_claimed: bool = False,
) -> dict:
    report_item_id = item_display_id(item)
    report_item_code = item_display_code(item)
    report_owner_name = str(getattr(item, "report_owner_name", "") or "").strip()
    report_owner_group = str(getattr(item, "report_owner_group", "") or "").strip()
    # Approved found items and matched/claimed lost items have a student-only
    # lifecycle. Unmatched lost reports still use the shared admin lifecycle.
    uses_personal_lifecycle = (
        item.status == "found"
        or (
            item.status == "lost"
            and (bool(item.is_matched) or bool(is_claimed))
        )
    )
    return {
        "id": item.id,
        "item_id": report_item_id,
        "item_code": report_item_code,
        "lost_id": report_item_code if item.status == "lost" else None,
        "found_id": report_item_code if item.status == "found" else None,
        "status": item.status,
        "category_id": item.category_id,
        "category": item.category,
        "item_name": item.item_name,
        "brand": item.brand,
        "color": item.color,
        "description": item.description,
        "location": item.location,
        "image_path": public_file_url(item.image_path),
        "date": item.date.isoformat() if item.date else None,
        "time_found": item.time_found,
        "is_matched": bool(item.is_matched),
        "is_claimed": bool(is_claimed),
        "is_surrendered": bool(item.is_surrendered),
        "archived": bool(
            item.student_archived if uses_personal_lifecycle else item.archived
        ),
        "deleted": bool(
            item.student_deleted if uses_personal_lifecycle else getattr(item, "deleted", False)
        ),
        "user_id": item.user_id,
        "report_owner_user_id": getattr(item, "report_owner_user_id", None),
        "uploader_name": report_owner_name or format_user_display_name(owner, "Self"),
        "report_owner_name": report_owner_name,
        "report_owner_group": report_owner_group,
    }


def serialize_student_pending_found(item: models.PendingItem, owner: models.User | None) -> dict:
    pending_code = format_item_code("pending_found", item.id)
    return {
        "id": item.id,
        "item_id": item.id,
        "item_code": pending_code,
        "found_id": pending_code,
        "status": "pending_found",
        "archived": bool(item.archived),
        "deleted": bool(getattr(item, "deleted", False)),
        "item_name": item.item_name,
        "category": item.category,
        "brand": item.brand,
        "color": item.color,
        "location": item.location,
        "image_path": public_file_url(item.image_path),
        "description": item.description,
        "date": item.date.isoformat() if item.date else None,
        "time_found": item.time_found,
        "uploader_name": format_user_display_name(owner, "Self"),
    }


@router.get("/notifications")
def get_student_notifications(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_active_student_user)
):
    notifications = db.query(models.Notification).filter(
        models.Notification.type.in_(["student_match", "student_update"]),
        models.Notification.related_id == current_user.id
    ).order_by(models.Notification.created_at.desc()).all()

    return notifications


@router.get("/notifications/unread-count")
def get_student_notification_unread_count(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_active_student_user)
):
    unread_count = db.query(models.Notification).filter(
        models.Notification.type.in_(["student_match", "student_update"]),
        models.Notification.related_id == current_user.id,
        models.Notification.is_read == False
    ).count()

    return {"unread_count": unread_count}


@router.post("/notifications/{notif_id}/read")
def mark_student_notification_read(
    notif_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_active_student_user)
):
    notif = db.query(models.Notification).filter(models.Notification.id == notif_id).first()
    if not notif:
        raise HTTPException(status_code=404, detail="Notification not found")

    is_owned_student_notif = notif.related_id == current_user.id and notif.type in {"student_match", "student_update"}
    if not is_owned_student_notif:
        raise HTTPException(status_code=403, detail="You do not have access to this notification")

    notif.is_read = True
    db.commit()
    return {"status": "success"}


@router.post("/notifications/mark-all-read")
def mark_all_student_notifications_read(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_active_student_user)
):
    db.query(models.Notification).filter(
        models.Notification.type.in_(["student_match", "student_update"]),
        models.Notification.related_id == current_user.id,
        models.Notification.is_read == False
    ).update({models.Notification.is_read: True}, synchronize_session=False)
    db.commit()
    return {"status": "success"}

@router.post("/found")
async def report_found_item(
    item_name: str = Form(...), # Added this to match your JS
    category_id: int = Form(None),
    category: str = Form(...),
    brand: str = Form(None),    # NEW: Brand
    color: str = Form(None),    # NEW: Color
    description: str = Form(None),
    location: str = Form(...),
    date: str = Form(None),
    time: str = Form(None),
    time_found: str = Form(None),
    image: UploadFile = File(...),
    extra_image_1: UploadFile = File(None),
    extra_image_2: UploadFile = File(None),
    image_embedding: str = Form(None),
    matched_item_id: int = Form(None),
    ai_score: float = Form(0.0),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_active_student_user)
):
    item_name = item_name.strip()
    if not item_name:
        raise HTTPException(status_code=400, detail="Item name is required")
    if len(item_name) > 255:
        raise HTTPException(status_code=400, detail="Item name must be 255 characters or fewer")

    # 2. Image Handling
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if image.content_type not in allowed_types:
        raise HTTPException(400, detail="Invalid image type")

    for upload, label in (
        (image, "Main image"),
        (extra_image_1, "Optional image 2"),
        (extra_image_2, "Optional image 3"),
    ):
        try:
            validate_upload_file_size(upload, label=label)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        resolved_category = resolve_category_name(db, category_id=category_id, category_name=category)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    query_images = []
    for upload in (image, extra_image_1, extra_image_2):
        if not upload or not upload.filename:
            continue
        if upload.content_type not in allowed_types:
            raise HTTPException(400, detail="Invalid image type")

        image_bytes = await upload.read()
        await upload.seek(0)
        if not image_bytes:
            continue

        try:
            query_images.append(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail="One of the uploaded images is invalid.") from exc

    db_path = save_file(image, resolved_category)
    parsed_date = None
    if date and date.strip():
        try:
            parsed_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.") from exc

    is_auto_match = ai_score >= 0.55 and matched_item_id is not None
    computed_embedding = image_embedding or ""
    if query_images:
        computed_embedding = json.dumps(
            (await run_in_threadpool(get_multi_image_embedding, query_images)).tolist()
        )

    resolved_time_found = (time_found or time or "").strip() or None

    # 3. SAVE TO PENDING TABLE
    pending_item = models.PendingItem(
        item_name=item_name,    # Added
        category=category,
        brand=brand,            # NEW: Saved here
        color=color,            # NEW: Saved here
        description=description,
        location=location,
        date=parsed_date,
        time_found=resolved_time_found,
        image_path=db_path,
        image_embedding=computed_embedding,
        matched_item_id=matched_item_id,
        user_id=current_user.id,
        created_at=datetime.utcnow(),
        archived=False
    )

    db.add(pending_item)
    db.flush()

    matched_lost_item = None
    matched_lost_owner = None
    match_score = float(ai_score or 0)
    if is_auto_match:
        matched_lost_item = db.query(models.Item).filter(
            models.Item.id == matched_item_id,
            models.Item.status == "lost",
            models.Item.archived == False
        ).first()
        if matched_lost_item:
            admin_match_score = f"{ai_score * 100:.1f}%"
            pending_item.matched_item_id = matched_lost_item.id
            matched_lost_item.is_matched = True
            possible_match_count = prepend_lost_possible_match(
                matched_lost_item,
                serialize_pending_found_match(pending_item, match_score)
            )
            db.add(models.Notification(
                message=f"AI MATCH ({admin_match_score}): Found {category} may match Lost Item #{matched_item_id}.",
                type="match",
                related_id=pending_item.id,
                target_url="/admin/Found_Items_Report",
                is_read=False,
                created_at=datetime.utcnow()
            ))

            owner_id = matched_lost_item.report_owner_user_id or matched_lost_item.user_id
            if owner_id:
                matched_lost_owner = db.query(models.User).filter(models.User.id == owner_id).first()
                reporter_name = current_user.full_name or current_user.email or "A student"
                create_student_notification(
                    db,
                    owner_id,
                    f"New possible match found: {reporter_name} submitted a found {category} that may match your lost item. You now have {possible_match_count} possible match(es). It is waiting for admin approval.",
                    "student_match",
                    f"/student/Lost-report?item_id={matched_lost_item.id}&show_match=1"
                )
    else:
        db.add(models.Notification(
            message=f"New Found Report: {category} ({item_name}) submitted by {current_user.full_name or current_user.email}.",
            type="new_report",
            related_id=pending_item.id,
            target_url="/admin/Found_Items_Report",
            is_read=False,
            created_at=datetime.utcnow()
        ))

    db.commit()
    db.refresh(pending_item)

    if matched_lost_item and matched_lost_owner:
        queue_lost_item_match_email(
            matched_lost_owner,
            matched_lost_item,
            message_text=(
                f"A found {category} may match your lost item. "
                "The report is waiting for admin approval."
            ),
        )

    return {
        "message": "Item reported successfully",
        "item_id": pending_item.id,
        "status": "pending_approval",
        "is_matched": False,
        "has_possible_match": bool(matched_lost_item)
    }

# Create a dedicated route for students
@router.post("/update-profile")
async def update_student_profile(
    full_name: str = Form(None),
    student_no: str = Form(None),
    course: str = Form(None),
    section: str = Form(None),
    profile_img: UploadFile = File(None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_active_student_user) 
):
    user = current_user
    
    if not user:
        return {"error": "User not found"}

    # Handle Image Upload
    if profile_img and profile_img.filename:
        try:
            validate_upload_file_size(profile_img, label="Profile image")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        user.profile_pic = save_file(profile_img, "profile-pics")

    # Only update fields that were submitted. Image-only saves should not erase
    # existing profile information.
    if full_name is not None:
        user.full_name = full_name
    if student_no is not None:
        user.student_no = student_no
    if course is not None:
        user.course = course
    if section is not None:
        user.section = section

    db.commit()
    create_student_notification(
        db,
        user.id,
        "Your student profile was updated successfully.",
        "student_update",
        "/student/profile"
    )
    db.commit()
    return {"message": "Student profile updated successfully"}

@router.get("/dashboard")
def Student_dashboard(
    request: Request,
    response: Response
):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"

    # REMOVED "admin": current_admin from the dictionary below
    return templates.TemplateResponse(
        "student2.0.html",
        {"request": request} 
    )

@router.get("/Messages")
def Student_messages(
    request: Request,
    response: Response
):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"

    return templates.TemplateResponse(
        "Student Pages/Student_Messages.html",
        {"request": request} 
    )
@router.get("/Lost-report")
def report_lost_item(
    request: Request,
    response: Response
):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"

    return templates.TemplateResponse(
        "Student Pages/Student_LostReport.html",
        {"request": request} 
    )
@router.get("/Found-report")
def report_found_item(
    request: Request,
    response: Response
):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"

    return templates.TemplateResponse(
        "Student Pages/Student_FoundReport.html",
        {"request": request} 
    )

@router.get("/profile")
def view_profile(
    request: Request,
    response: Response
):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"

    return templates.TemplateResponse(
        "Student Pages/Student_profile.html",
        {"request": request} 
    )

@router.get("/settings")
def view_settings(
    request: Request,
    response: Response
):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"

    return templates.TemplateResponse(
        "Student Pages/Student_Settings.html",
        {"request": request} 
    )


@router.post("/update-settings")
def update_student_settings(
    data: SettingsUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_active_student_user),
):
    user = db.query(models.User).filter(models.User.id == current_user.id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.two_factor_enabled = bool(data.two_factor)
    user.push_notifications = bool(data.notifications)
    user.theme_mode = (data.theme or "light")[:20]
    user.font_size = max(12, min(24, int(data.font_size)))
    db.commit()
    create_student_notification(
        db,
        user.id,
        "Your student settings were updated successfully.",
        "student_update",
        "/student/settings"
    )
    db.commit()

    return {"status": "success", "message": "Student settings updated successfully"}


# ... (your existing imports)
@router.get("/items/found/me")
def get_my_found_items(
    view: str = "active",
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_active_student_user)
):
    view = view if view in {"active", "archive", "deleted"} else "active"
    lifecycle_filters = {
        "active": (False, False),
        "archive": (True, False),
        "deleted": (True, True),
    }
    archived, deleted = lifecycle_filters[view]
    pending = db.query(models.PendingItem).filter(
        models.PendingItem.user_id == current_user.id,
        models.PendingItem.archived == archived,
        models.PendingItem.deleted == deleted,
    ).all()
    
    approved = db.query(models.Item).filter(
        models.Item.user_id == current_user.id,
        models.Item.status == "found",
        models.Item.archived == False,
        models.Item.deleted == False,
        models.Item.student_archived == archived,
        models.Item.student_deleted == deleted,
        models.Item.student_hidden == False,
    ).all()
    approved_ids = [item.id for item in approved]
    claimed_found_ids = set()
    if approved_ids:
        claimed_found_ids = {
            found_id for (found_id,) in db.query(models.Claim.found_item_id).filter(
                models.Claim.found_item_id.in_(approved_ids),
                models.Claim.status.in_(models.CLAIMED_CLAIM_STATUSES),
            ).all()
        }

    results = []
    
    for p in pending:
        results.append({
            "display_status": "Pending Approval",
            "data": serialize_student_pending_found(p, current_user)
        })
        
    for a in approved:
        is_claimed = a.id in claimed_found_ids
        display_status = "Claimed Item" if is_claimed else ("Matched" if a.is_matched else "Approved")
        results.append({
            "display_status": display_status,
            "data": serialize_student_item(a, current_user, is_claimed=is_claimed)
        })
        
    return results


@router.get("/items/found/active-count")
def get_active_found_item_count(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_active_student_user)
):
    count = db.query(models.Item).filter(
        models.Item.status == "found",
        models.Item.archived == False
    ).count()
    return {"count": count}


@router.put("/items/pending-found/{item_id}/edit")
async def edit_pending_found_item(
    item_id: int,
    item_name: str = Form(...),
    category: str = Form(...),
    brand: str = Form(None),
    color: str = Form(None),
    description: str = Form(None),
    location: str = Form(...),
    date: str = Form(None),
    time_found: str = Form(None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_active_student_user)
    image: UploadFile = File(None),
    extra_image_1: UploadFile = File(None),
    extra_image_2: UploadFile = File(None),
):
    item = db.query(models.PendingItem).filter(
        models.PendingItem.id == item_id,
        models.PendingItem.user_id == current_user.id
    ).first()

    if not item:
        raise HTTPException(
            status_code=404,
            detail="Pending item not found"
        )

    item.item_name = item_name
    item.category = category
    item.brand = brand
    item.color = color
    item.description = description
    item.location = location

    if date:
        item.date = datetime.strptime(
            date,
            "%Y-%m-%d"
        ).date()

    item.time_found = time_found

    db.commit()

    return {
        "message": "Pending item updated successfully"
    }

@router.put("/items/lost/{item_id}/edit")
async def edit_lost_item(
    item_id: int,
    item_name: str = Form(...),
    category: str = Form(...),
    category_id: int = Form(None),
    brand: str = Form(None),
    color: str = Form(None),
    description: str = Form(None),
    location: str = Form(...),
    date: str = Form(None),
    time_found: str = Form(None),
    image: UploadFile = File(None),
    extra_image_1: UploadFile = File(None),
    extra_image_2: UploadFile = File(None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_active_student_user)
):
    item = db.query(models.Item).filter(
        models.Item.id == item_id,
        models.Item.status == "lost",
        models.Item.user_id == current_user.id
    ).first()

    if not item:
        raise HTTPException(
            status_code=404,
            detail="Lost item not found"
        )

    # Only pending-match lost items can be edited
    if not item.is_matched:
        raise HTTPException(
            status_code=400,
            detail="Only pending-match lost items can be edited"
        )

    # Optional: prevent editing if already claimed
    existing_claim = db.query(models.Claim).filter(
        models.Claim.lost_item_id == item.id,
        models.Claim.status.in_(models.CLAIMED_CLAIM_STATUSES)
    ).first()

    if existing_claim:
        raise HTTPException(
            status_code=400,
            detail="Claimed items can no longer be edited"
        )

    item.item_name = item_name.strip()
    item.category = category

    if category_id is not None:
        item.category_id = category_id

    item.brand = brand
    item.color = color
    item.description = description
    item.location = location
    item.time_found = time_found

    if date and date.strip():
        try:
            item.date = datetime.strptime(
                date,
                "%Y-%m-%d"
            ).date()
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid date format. Use YYYY-MM-DD."
            )

    db.commit()
    db.refresh(item)

    return {
        "status": "success",
        "message": "Lost item updated successfully",
        "item_id": item.id
    }
    
@router.post("/items/lost/report")
async def submit_user_lost_report(
    item_name: str = Form(...),
    category: str = Form(...),
    category_id: int = Form(...),
    location: str = Form(...),
    description: str = Form(None),
    brand: str = Form(None),
    color: str = Form(None),
    date: str = Form(None),
    time_found: str = Form(None),
    image: UploadFile = File(None),
    extra_image_1: UploadFile = File(None),
    extra_image_2: UploadFile = File(None),
    image_embedding: str = Form(None), # Embedding from frontend AI call
    possible_matches: str = Form(None),
    matched_item_id: int = Form(None), # Found ID if user confirmed a match
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_active_student_user)
):
    item_name = item_name.strip()
    if not item_name:
        raise HTTPException(status_code=400, detail="Item name is required")
    if len(item_name) > 255:
        raise HTTPException(status_code=400, detail="Item name must be 255 characters or fewer")

    # 1. Handle the image upload using your save_file helper
    saved_path = None
    query_images = []

    for upload, label in (
        (image, "Main image"),
        (extra_image_1, "Optional image 2"),
        (extra_image_2, "Optional image 3"),
    ):
        try:
            validate_upload_file_size(upload, label=label)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if image and image.filename:
        image_bytes = await image.read()
        await image.seek(0)
        if image_bytes:
            try:
                query_images.append(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Invalid primary image upload.") from exc

        try:
            resolved_category = resolve_category_name(db, category_id=category_id, category_name=category)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        saved_path = save_file(image, resolved_category)

    for extra_upload in (extra_image_1, extra_image_2):
        if not extra_upload or not extra_upload.filename:
            continue
        extra_bytes = await extra_upload.read()
        await extra_upload.seek(0)
        if not extra_bytes:
            continue
        try:
            query_images.append(Image.open(io.BytesIO(extra_bytes)).convert("RGB"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail="One of the optional images is invalid.") from exc

    computed_embedding = image_embedding or ""
    if query_images:
        computed_embedding = json.dumps(
            (await run_in_threadpool(get_multi_image_embedding, query_images)).tolist()
        )

    # 2. Date Parsing
    parsed_date = None
    if date and date.strip():
        try:
            parsed_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            parsed_date = None

    # 3. Create the Item Record
    saved_possible_matches = normalize_saved_possible_matches(possible_matches)

    is_auto_match = matched_item_id is not None

    new_report = models.Item(
        status="lost",
        item_name=item_name.strip(),
        category_id=category_id,
        category=category,
        brand=brand,
        color=color,
        description=description,
        location=location,
        image_path=saved_path,
        image_embedding=computed_embedding,
        possible_matches=saved_possible_matches,
        user_id=current_user.id,
        date=parsed_date,
        time_found=(time_found or "").strip() or None,
        is_matched=is_auto_match,
        department=None, # Explicitly no department for student reports
        is_surrendered=False, # Students keep their lost item (it's lost!)
        created_at=datetime.utcnow()
    )

    try:
        db.add(new_report)
        db.flush() # Generate new_report.id for relationships

        # 4. Handle possible match and notification logic
        if matched_item_id:
            if not current_user.student_no:
                raise HTTPException(
                    status_code=400,
                    detail="Your account does not have a student number yet. Please update your profile or contact an admin before claiming an item."
                )

            found_item = db.query(models.Item).filter(models.Item.id == matched_item_id).first()
            if not found_item or found_item.status != "found" or found_item.archived:
                raise HTTPException(status_code=404, detail="Selected possible match is no longer available.")

            new_report.is_matched = True
            found_item.is_matched = True
            new_claim = ensure_student_claim_for_pair(
                db,
                lost_item=new_report,
                found_item=found_item,
                claimant_id=current_user.id
            )

            # Notification for Admin (Match)
            notif = models.Notification(
                message=f"Possible AI match: {current_user.full_name} reported a lost item that may match Item #{matched_item_id}",
                type="match",
                related_id=new_claim.id,
                target_url=f"/admin/Reports?report_type=claim&claim_id={new_claim.id}",
                is_read=False
            )
            db.add(notif)
        else:
            # Notification for Admin (New General Report)
            notif = models.Notification(
                message=f"New Lost Report: {category} ({item_name}) from {current_user.full_name}",
                type="new_report",
                related_id=new_report.id,
                target_url="/admin/Lost_Items_Report",
                is_read=False
            )
            db.add(notif)

        db.commit()
        db.refresh(new_report)

        if new_report.is_matched:
            queue_lost_item_match_email(
                current_user,
                new_report,
                subject="A match was found for your lost item",
                message_text=f"A found {category} was matched with your lost-item report.",
            )
        
        return {
            "status": "success", 
            "item_id": new_report.id,
            "is_matched": bool(new_report.is_matched),
            "has_possible_match": bool(matched_item_id)
        }

    except Exception as e:
        db.rollback()
        print(f"Database Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit report")
@router.get("/api/items/lost/me")
def get_my_lost_reports(
    view: str = "active",
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_active_student_user)
):
    view = view if view in {"active", "archive", "deleted"} else "active"
    archived, deleted = {
        "active": (False, False),
        "archive": (True, False),
        "deleted": (True, True),
    }[view]
    current_user_name = format_user_display_name(current_user, "").strip().lower()
    # Change 'LostItem' to 'Item' to match your models.py
    reports = db.query(models.Item).filter(
        or_(
            models.Item.user_id == current_user.id,
            models.Item.report_owner_user_id == current_user.id,
            and_(
                models.Item.report_owner_user_id.is_(None),
                func.lower(models.Item.report_owner_name) == current_user_name,
            ) if current_user_name else False,
        ),
        models.Item.status == "lost",
    ).all()

    report_ids = [report.id for report in reports]
    claimed_lost_ids = set()
    if report_ids:
        claimed_lost_ids = {
            lost_item_id for (lost_item_id,) in db.query(models.Claim.lost_item_id).filter(
                models.Claim.lost_item_id.in_(report_ids),
                models.Claim.status.in_(models.CLAIMED_CLAIM_STATUSES),
            ).all()
        }

    results = []
    for report in reports:
        is_claimed = report.id in claimed_lost_ids
        uses_personal_lifecycle = bool(report.is_matched) or is_claimed
        if uses_personal_lifecycle:
            belongs_in_view = (
                not bool(report.archived)
                and not bool(report.deleted)
                and not bool(report.student_hidden)
                and bool(report.student_archived) == archived
                and bool(report.student_deleted) == deleted
            )
        else:
            belongs_in_view = (
                bool(report.archived) == archived
                and bool(report.deleted) == deleted
            )
        if not belongs_in_view:
            continue
        item_data = serialize_student_item(
            report,
            current_user,
            is_claimed=is_claimed,
        )
        item_data["display_status"] = (
            "Claimed Item"
            if is_claimed
            else ("Matched" if report.is_matched else "Pending")
        )
        results.append(item_data)

    return results


def get_owned_student_report(db: Session, current_user: models.User, record_type: str, item_id: int):
    if record_type == "pending-found":
        return db.query(models.PendingItem).filter(
            models.PendingItem.id == item_id,
            models.PendingItem.user_id == current_user.id,
        ).first()
    if record_type not in {"found", "lost"}:
        return None
    return db.query(models.Item).filter(
        models.Item.id == item_id,
        models.Item.status == record_type,
        or_(
            models.Item.user_id == current_user.id,
            models.Item.report_owner_user_id == current_user.id,
        ),
    ).first()


def uses_student_personal_lifecycle(
    db: Session,
    item,
    record_type: str,
) -> bool:
    """Return whether a student action must leave the admin record unchanged."""
    if record_type == "found":
        return True
    if record_type != "lost":
        return False
    if bool(getattr(item, "is_matched", False)):
        return True
    return db.query(models.Claim.id).filter(
        models.Claim.lost_item_id == item.id,
        models.Claim.status.in_(models.CLAIMED_CLAIM_STATUSES),
    ).first() is not None


@router.put("/items/{record_type}/{item_id}/archive")
def archive_my_report(
    record_type: str,
    item_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_active_student_user),
):
    item = get_owned_student_report(db, current_user, record_type, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Report not found")
    use_personal_lifecycle = uses_student_personal_lifecycle(db, item, record_type)
    if use_personal_lifecycle:
        item.student_archived = True
        item.student_deleted = False
    else:
        item.archived = True
        item.deleted = False
    db.commit()
    return {"message": "Report archived"}


@router.put("/items/{record_type}/{item_id}/delete")
def delete_my_report(
    record_type: str,
    item_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_active_student_user),
):
    item = get_owned_student_report(db, current_user, record_type, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Report not found")
    use_personal_lifecycle = uses_student_personal_lifecycle(db, item, record_type)
    if use_personal_lifecycle:
        item.student_archived = True
        item.student_deleted = True
        message = "Report moved to your Deleted Items; administrator inventory unchanged"
    else:
        # An unapproved found upload is still a shared PendingItem, so deleting
        # it must also remove it from the administrator's active pending queue.
        item.archived = True
        item.deleted = True
        message = "Pending report moved to Deleted Items for both student and administrator"
    db.commit()
    return {"message": message}


@router.put("/items/{record_type}/{item_id}/recover")
def recover_my_report(
    record_type: str,
    item_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_active_student_user),
):
    item = get_owned_student_report(db, current_user, record_type, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Report not found")
    use_personal_lifecycle = uses_student_personal_lifecycle(db, item, record_type)
    if use_personal_lifecycle:
        item.student_archived = False
        item.student_deleted = False
    else:
        item.archived = False
        item.deleted = False
    db.commit()
    return {"message": "Report recovered"}


@router.delete("/items/{record_type}/{item_id}/permanent")
def permanently_delete_my_report(
    record_type: str,
    item_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_active_student_user),
):
    item = get_owned_student_report(db, current_user, record_type, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Deleted report not found")
    use_personal_lifecycle = uses_student_personal_lifecycle(db, item, record_type)
    if use_personal_lifecycle:
        if not bool(getattr(item, "student_deleted", False)):
            raise HTTPException(status_code=404, detail="Deleted report not found")
        item.student_archived = False
        item.student_deleted = False
        item.student_hidden = True
        db.commit()
        return {"message": "Report removed from your items"}
    if not bool(getattr(item, "deleted", False)):
        raise HTTPException(status_code=404, detail="Deleted report not found")
    if record_type != "pending-found":
        linked_claims = db.query(models.Claim).filter(or_(
            models.Claim.lost_item_id == item_id,
            models.Claim.found_item_id == item_id,
        )).all()
        if linked_claims:
            claim_ids = [claim.id for claim in linked_claims]
            db.query(models.ClaimDecisionReport).filter(
                models.ClaimDecisionReport.claim_id.in_(claim_ids)
            ).delete(synchronize_session=False)
            db.query(models.ClaimProof).filter(
                models.ClaimProof.claim_id.in_(claim_ids)
            ).delete(synchronize_session=False)
            db.query(models.Claim).filter(
                models.Claim.id.in_(claim_ids)
            ).delete(synchronize_session=False)
    db.delete(item)
    db.commit()
    return {
        "message": (
            "Pending report permanently deleted for both student and administrator"
            if record_type == "pending-found"
            else "Report permanently deleted"
        )
    }

# --- Your existing page routes below ---

@router.get("/Lost-report")
def report_lost_item(request: Request, response: Response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return templates.TemplateResponse("Student Pages/Student_LostReport.html", {"request": request})

# ... (rest of your routes)
