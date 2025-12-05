from pydantic import BaseModel, Field


class CandidateFeatures(BaseModel):
    Age: float
    Gender: str
    Education_Level: str = Field(alias="Education Level")
    Job_Title: str = Field(alias="Job Title")
    Years_of_Experience: float = Field(alias="Years of Experience")

    class Config:
        populate_by_name = True        # allow using Python names
        json_schema_extra = {
            "example": {
                "Age": 32,
                "Gender": "Male",
                "Education Level": "Bachelor's",
                "Job Title": "Software Engineer",
                "Years of Experience": 5
            }
        }
