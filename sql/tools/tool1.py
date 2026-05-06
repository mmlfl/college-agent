from sqlalchemy.orm import Session
from langchain_core.tools import tool
from sql.table.models import Booking, Venue, engine


@tool
def query_venues(question: str = "") -> str:
    """查询所有场地信息。参数: 用户问题(可选)"""
    with Session(engine) as session:
        venues = session.query(Venue).all()
        result = "\n".join(
            f"- {v.name} ({v.type}) | 位置: {v.location} | ID: {v.id}"
            for v in venues
        )
        return result


@tool
def check_availability(venue_name: str, date: str) -> str:
    """查询某场地在某天的预约情况。参数: venue_name(场地名,支持模糊匹配), date(YYYY-MM-DD)"""
    with Session(engine) as session:
        venue = session.query(Venue).filter(
            Venue.name.like(f"%{venue_name}%")
        ).first()
        if not venue:
            return f"未找到名为 '{venue_name}' 的场地"

        bookings = session.query(Booking).filter(
            Booking.venue_id == venue.id,
            Booking.start_time.like(f"{date}%"),
            Booking.status != "cancelled"
        ).all()
        if not bookings:
            return f"场地 {venue.name}(ID:{venue.id}) 在 {date} 全天空闲|venue_id:{venue.id}"
        occupied = "; ".join(f"{b.start_time}~{b.end_time}" for b in bookings)
        return f"场地 {venue.name}(ID:{venue.id}) 在 {date} 有以下预约: {occupied}|venue_id:{venue.id}"


@tool
def create_booking(student_id: int, venue_id: int, start: str, end: str) -> str:
    """创建预约。参数: 学生ID, 场地ID, 开始时间(YYYY-MM-DD HH:MM), 结束时间"""
    with Session(engine) as session:
        # 检查冲突
        conflicts = session.query(Booking).filter(
            Booking.venue_id == venue_id,
            Booking.start_time < end,
            Booking.end_time > start,
            Booking.status != "cancelled"
        ).all()
        if conflicts:
            return "该时间段已被预约，请选择其他时间"
        booking = Booking(
            student_id=student_id,
            venue_id=venue_id,
            start_time=start,
            end_time=end,
            status="confirmed"
        )
        session.add(booking)
        session.commit()
        return f"预约成功！预约ID: {booking.id}"


@tool
def cancel_booking(booking_id: int) -> str:
    """取消预约。参数: 预约ID"""
    with Session(engine) as session:
        booking = session.query(Booking).filter(Booking.id == booking_id).one_or_none()
        if not booking:
            return f"未找到预约ID为 {booking_id} 的记录"
        if booking.status == "cancelled":
            return "该预约已取消"
        booking.status = "cancelled"
        session.commit()
        return f"预约 {booking_id} 已取消"