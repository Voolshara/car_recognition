import os 
import datetime
import logging
from dotenv import load_dotenv
from sqlalchemy import ForeignKey, String, and_
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import select, update
from sqlalchemy import bindparam


logging.basicConfig(
    # filename="logs",
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


class Base(DeclarativeBase):
    pass


class Vehicles(Base):
    __tablename__ = "DetectedVehicles"
    Id: Mapped[int] = mapped_column(primary_key=True)
    Vehicles_number: Mapped[str]
    Time_in: Mapped[datetime.datetime]
    Time_out: Mapped[datetime.datetime]

    def __repr__(self) -> str:
        return f"Name(id={self.id!r}, car={self.Vehicles_number!r}, in={self.Time_in!r}, out={self.Time_out!r})"


class DB_worker():
    def __init__(self) -> None:

        load_dotenv()
        self.engine = create_engine(
            'mssql+pyodbc://{}:{}@{}/{}'.format(
                os.getenv('POSTGRES_USER'),
                os.getenv('POSTGRES_PASSWORD'),
                os.getenv('POSTGRES_HOST'),
                # os.getenv('POSTGRES_PORT'),
                os.getenv('POSTGRES_DB'),
            )
        )


    def set_car(self, Vehicles_number: str):
        session = Session(self.engine)
        car_query = select(Vehicles).where(
            Vehicles.Vehicles_number == Vehicles_number
        )
        fetch_data = session.scalars(car_query).first()
        if fetch_data is None:
            with Session(self.engine) as session:
                Vehicle = Vehicles(
                    Vehicles_number=Vehicles_number,
                    Time_in = datetime.datetime.now(),
                    Time_out = datetime.datetime.now(),
                )
                session.add_all([Vehicle])
                session.commit()
        else:
            car_id = fetch_data.Id
            with Session(self.engine) as session:
                session.query(Vehicles).filter(Vehicles.Id == car_id).update({"Time_out" : datetime.datetime.now()})
                session.commit()

    def create_tables(self):
        Base.metadata.create_all(self.engine)


def create_tables():
    DBW = DB_worker()
    DBW.create_tables()