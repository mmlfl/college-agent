from entity.bookingEntity import BookingEntity
from fastapi import APIRouter
from sqlalchemy.orm import Session
from booking.models import engine, Booking

router = APIRouter()

@router.get("/booking",response_model=list[BookingEntity])
async def get_bookings():
    with Session(engine) as session:
        bookings = session.query(Booking).all()
        return bookings

@router.get("/booking/{id}",response_model=BookingEntity)
async def get_booking(id:int):
    with Session(engine) as session:
        booking = session.query(Booking).filter(Booking.id == id).one_or_none()
        return booking

@router.post("/booking")
async def create_booking(booking: BookingEntity):
    with Session(engine) as session:
        conflicts = session.query(Booking).filter(
            Booking.venue_id == booking.venue_id,
            Booking.start_time < booking.end_time,
            Booking.end_time > booking.start_time,
            Booking.status != "cancelled"
        ).all()
        if(conflicts):
            return "该场地在该时间段已被预约,请预约其他时间段,或预约其他场地"
        book = Booking(
            student_id=booking.student_id,
            venue_id=booking.venue_id,
            start_time=booking.start_time,
            end_time=booking.end_time,
            status="confirmed"
        )
        session.add(book)
        session.commit()
        session.refresh(book)
        return {f"{book.id}": book}

@router.delete("/booking/{id}")
async def delete_booking(id:int):
    """取消预约"""
    with Session(engine) as session:
        booking = session.query(Booking).filter(Booking.id == id).one_or_none()
        if booking is None:
            return {"message": "未找到该预约"}
        if booking.status == "cancelled":
            return {"message": "该预约已取消"}
        booking.status = "cancelled"
        session.commit()
        session.refresh(booking)
        return {"message": f"Booking {id} deleted"}
