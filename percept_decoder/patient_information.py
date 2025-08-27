from percept_decoder.base_schema import BaseSchema


class PatientInformation(BaseSchema):
    patient_first_name: str
    patient_last_name: str
    patient_gender: str
    patient_date_of_birth: str
    patient_id: str
    clinician_notes: str
    diagnosis: str  # TODO: this is some typedef
