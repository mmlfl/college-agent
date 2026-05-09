from fastapi import APIRouter
from sqlalchemy.orm import Session
from graph.sql.sql_previous_work.entity.venueEntity import VenueEntity

from graph.sql.sql_previous_work.table.models import Venue, engine

router = APIRouter()


@router.get("/venue/{id}",response_model=VenueEntity)
async def get_venue(id: int):
    """Get a venue by its id."""
    with Session(engine) as session:
        venue = session.query(Venue).filter(Venue.id == id).one_or_none()
        return venue


@router.get("/venues",response_model=list[VenueEntity])
async def get_venues():
    """Get all venues."""
    with Session(engine) as session:
        venues = session.query(Venue).all()
        return venues


@router.post("/venues")
async def create_venue(venue_in: VenueEntity):
    """Create a new venue."""
    with Session(engine) as session:
        venue = Venue(
            name=venue_in.name,
            type=venue_in.type,
            location=venue_in.location,
        )
        session.add(venue)
        session.commit()
        session.refresh(venue)
        return {f"{venue.id}": venue}
