from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import anthropic
import httpx
import json
import os
from datetime import datetime

app = FastAPI(title="ClimateNet", description="AI-powered climate intervention modeling and carbon capture optimization agent")
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

class ClimateQuery(BaseModel):
  region: str
  intervention_type: str
  target_year: int
  budget_usd_millions: Optional[float] = None

class ClimateIntervention(BaseModel):
  region: str
  intervention: str
  projected_co2_reduction_tons: float
  cost_per_ton_usd: float
  implementation_timeline_years: int
  co_benefits: List[str]
  risks: List[str]
  confidence: float
  data_sources: List[str]

class CarbonQuery(BaseModel):
  technology: str
  scale: str
  location: str

class CarbonAnalysis(BaseModel):
  technology: str
  capture_rate_tons_per_year: float
  energy_requirement_mwh: float
  land_use_hectares: float
  cost_usd_per_ton: float
  maturity_level: str
  recommendations: str

CLIMATE_DATA = {
  "global_temp_increase_c": 1.2,
  "annual_co2_emissions_gt": 37.4,
  "sea_level_rise_mm_per_year": 3.7,
  "arctic_ice_loss_percent_per_decade": 13,
  "interventions": {
    "reforestation": {"cost_per_ton": 15, "maturity": "proven"},
    "direct_air_capture": {"cost_per_ton": 250, "maturity": "emerging"},
    "ocean_iron_fertilization": {"cost_per_ton": 20, "maturity": "experimental"},
    "enhanced_weathering": {"cost_per_ton": 50, "maturity": "pilot"},
    "solar_geoengineering": {"cost_per_ton": 5, "maturity": "theoretical"},
    "blue_carbon": {"cost_per_ton": 30, "maturity": "proven"},
  }
}

async def fetch_climate_data(region: str) -> dict:
  async with httpx.AsyncClient() as hclient:
    try:
      url = "https://api.open-meteo.com/v1/forecast"
      params = {"latitude": 40.7128, "longitude": -74.0060, "current": "temperature_2m,precipitation", "forecast_days": 1}
      r = await hclient.get(url, params=params, timeout=10)
      weather = r.json()
      return {"region": region, "current_conditions": weather.get("current", {}), "timestamp": datetime.utcnow().isoformat()}
    except:
      return {"region": region, "status": "data unavailable"}

async def fetch_co2_data() -> dict:
  async with httpx.AsyncClient() as hclient:
    try:
      url = "https://global-warming.org/api/co2-api"
      r = await hclient.get(url, timeout=10)
      data = r.json()
      co2_readings = data.get("co2", [])
      if co2_readings:
        latest = co2_readings[-1]
        return {"co2_ppm": latest.get("trend", "unknown"), "year": latest.get("year", "unknown")}
      return {"co2_ppm": 421, "year": 2024}
    except:
      return {"co2_ppm": 421, "year": 2024}

@app.post("/analyze-intervention", response_model=ClimateIntervention)
async def analyze_climate_intervention(query: ClimateQuery):
  climate_data = await fetch_climate_data(query.region)
  co2_data = await fetch_co2_data()
  intervention_info = CLIMATE_DATA["interventions"].get(query.intervention_type.lower().replace(" ", "_"), {})

  prompt = f"""You are a climate science AI agent analyzing intervention strategies.

Region: {query.region}
Intervention: {query.intervention_type}
Target Year: {query.target_year}
Budget: ${query.budget_usd_millions}M USD
Current CO2: {co2_data.get('co2_ppm')} ppm
Region Climate Data: {climate_data}
Intervention Reference Data: {intervention_info}
Global Stats: {CLIMATE_DATA}

Analyze this intervention and provide:
1. Projected CO2 reduction in metric tons
2. Cost per ton USD
3. Implementation timeline in years
4. Co-benefits (biodiversity, jobs, water, etc.)
5. Key risks and uncertainties
6. Confidence score 0.0-1.0
7. Key data sources

Respond as JSON:
{{
  "projected_co2_reduction_tons": 0.0,
  "cost_per_ton_usd": 0.0,
  "implementation_timeline_years": 0,
  "co_benefits": ["benefit1"],
  "risks": ["risk1"],
  "confidence": 0.0,
  "data_sources": ["source1"]
}}"""

  response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1200,
    messages=[{"role": "user", "content": prompt}]
  )
  text = response.content[0].text
  start = text.find("{")
  end = text.rfind("}") + 1
  try:
    result = json.loads(text[start:end])
  except json.JSONDecodeError:
    result = {"projected_co2_reduction_tons": 0, "cost_per_ton_usd": 100, "implementation_timeline_years": 10, "co_benefits": [], "risks": [text], "confidence": 0.4, "data_sources": []}

  return ClimateIntervention(
    region=query.region,
    intervention=query.intervention_type,
    **{k: result.get(k, v) for k, v in {
      "projected_co2_reduction_tons": 0.0, "cost_per_ton_usd": 100.0,
      "implementation_timeline_years": 10, "co_benefits": [], "risks": [],
      "confidence": 0.5, "data_sources": []
    }.items()}
  )

@app.post("/carbon-capture-analysis", response_model=CarbonAnalysis)
async def analyze_carbon_capture(query: CarbonQuery):
  co2_data = await fetch_co2_data()

  prompt = f"""You are a carbon capture technology AI analyst.

Technology: {query.technology}
Scale: {query.scale}
Location: {query.location}
Current CO2: {co2_data} ppm

Analyze this carbon capture approach as JSON:
{{"capture_rate_tons_per_year": 0.0, "energy_requirement_mwh": 0.0, "land_use_hectares": 0.0, "cost_usd_per_ton": 0.0, "maturity_level": "pilot/proven/emerging", "recommendations": "key recommendations"}}"""

  response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=700,
    messages=[{"role": "user", "content": prompt}]
  )
  text = response.content[0].text
  start = text.find("{")
  end = text.rfind("}") + 1
  try:
    result = json.loads(text[start:end])
  except json.JSONDecodeError:
    result = {"capture_rate_tons_per_year": 1000, "energy_requirement_mwh": 500, "land_use_hectares": 100, "cost_usd_per_ton": 150, "maturity_level": "pilot", "recommendations": text}

  return CarbonAnalysis(technology=query.technology, **result)

@app.get("/climate-stats")
async def get_climate_stats():
  co2 = await fetch_co2_data()
  return {**CLIMATE_DATA, "live_co2": co2}

@app.get("/health")
def health():
  return {"status": "ok", "service": "climatenet"}

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
