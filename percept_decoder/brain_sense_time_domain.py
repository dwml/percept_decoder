from percept_decoder.base_schema import BaseSchema
from pydantic import Field


class BrainSenseTimeDomain(BaseSchema):
    pass_: str = Field(alias="Pass")
    global_sequences: str
    global_packet_sizes: str
    ticks_in_mses: str
    channel: str
    gain: int
    first_packet_date_time: str
    sample_rate_in_hz: int
    time_domain_data: list[float]
