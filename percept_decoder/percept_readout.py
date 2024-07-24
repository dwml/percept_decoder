from percept_decoder.base_schema import BaseSchema
from percept_decoder.brain_sense_time_domain import BrainSenseTimeDomain

from typing import Optional
from pydantic import Field


class PerceptReadOut(BaseSchema):
    """The BaseSchema provides pascal case aliasses, but for the optional fields,
    we need to provide a Field class give the alias explicitly, and provide a default
    value.
    """

    abnormal_end: bool
    fully_read_for_session: bool
    feature_information_code: str
    session_date: str
    session_end_data: Optional[str] = Field(default=None, alias="SessionEndDate")
    programmer_timezone: str
    programmer_utc_offset: str
    programmer_locale: str
    programmer_version: str
    patient_information: dict  # nice to have a PatientInformation object
    device_information: dict  # nice to have a DeviceInformation object
    battery_information: dict  # nice to have a BatteryInformation object
    lead_configuration: dict  # nice to have a LeadConfiguration object
    stimulation: dict
    groups: dict
    battery_reminder: dict
    most_recent_in_session_signal_check: list
    impedance: list
    annotations: Optional[list] = Field(default=None, alias="Annotations")
    group_history: list
    brain_sense_time_domain: list[BrainSenseTimeDomain]
    brain_sense_lfp: list[dict]
