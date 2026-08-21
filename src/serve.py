import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="MLOps Wine Quality Inference Service")

GCS_BUCKET = os.environ.get("GCS_BUCKET") or os.environ.get("CLOUD_BUCKET") or os.environ.get("S3_BUCKET", "")
MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser(os.environ.get("MODEL_LOCAL_PATH", "~/models/model.pkl"))
FALLBACK_LOCAL_PATH = "models/model.pkl"


def download_model():
    """
    Tai file model.pkl tu Cloud Storage (S3 / OCI / GCS) ve may khi server khoi dong.
    Neu khong co bien moi truong Cloud hoac da co model cuc bo, su dung file cuc bo.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # 1. Thu tai tu S3 / OCI Object Storage (neu co boto3 va AWS credentials/endpoint)
    s3_endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("S3_ENDPOINT_URL")
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if GCS_BUCKET and (aws_access_key or s3_endpoint):
        try:
            import boto3
            session = boto3.session.Session()
            s3_client = session.client(
                "s3",
                endpoint_url=s3_endpoint,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=os.environ.get("AWS_REGION", "us-east-1"),
            )
            s3_client.download_file(GCS_BUCKET, MODEL_KEY, MODEL_PATH)
            print(f"[SERVE] Model downloaded successfully from S3/OCI (bucket: {GCS_BUCKET}) to {MODEL_PATH}")
            return
        except Exception as e:
            print(f"[SERVE] S3 download attempt failed: {e}")

    # 2. Thu tai tu Google Cloud Storage (neu co GCS)
    if GCS_BUCKET and os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET)
            blob = bucket.blob(MODEL_KEY)
            blob.download_to_filename(MODEL_PATH)
            print(f"[SERVE] Model downloaded successfully from GCS (bucket: {GCS_BUCKET}) to {MODEL_PATH}")
            return
        except Exception as e:
            print(f"[SERVE] GCS download attempt failed: {e}")

    # 3. Fallback: Kiem tra neu model da co san tai MODEL_PATH hoac FALLBACK_LOCAL_PATH
    if os.path.exists(MODEL_PATH):
        print(f"[SERVE] Using existing model file at {MODEL_PATH}")
    elif os.path.exists(FALLBACK_LOCAL_PATH):
        print(f"[SERVE] Using local fallback model file at {FALLBACK_LOCAL_PATH}")
    else:
        print(f"[SERVE] Warning: No model file found yet at {MODEL_PATH} or {FALLBACK_LOCAL_PATH}")


# Khoi dong va load model khi module duoc nap
download_model()

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
elif os.path.exists(FALLBACK_LOCAL_PATH):
    model = joblib.load(FALLBACK_LOCAL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f12]}
    Dau ra  : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}

    Thu tu 12 dac trung:
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol, wine_type
    """
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        elif os.path.exists(FALLBACK_LOCAL_PATH):
            model = joblib.load(FALLBACK_LOCAL_PATH)
        else:
            raise HTTPException(status_code=503, detail="Model is not loaded or not found")

    if len(req.features) != 12:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 12 features (wine quality), got {len(req.features)}",
        )

    preds = model.predict([req.features])
    prediction = int(preds[0])

    label_mapping = {0: "thap", 1: "trung_binh", 2: "cao"}
    label = label_mapping.get(prediction, "unknown")

    return {"prediction": prediction, "label": label}


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "::")
    port = int(os.environ.get("PORT", "80"))
    print(f"[SERVE] Starting server on [{host}]:{port} (IPv4 + IPv6 dual-stack)")
    uvicorn.run(app, host=host, port=port)
