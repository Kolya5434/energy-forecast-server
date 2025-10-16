from fastapi import FastAPI

app = FastAPI(
    title="Development of a hybrid energy consumption forecasting system for urban networks using classical and modern ML models",
    description="Розробка гібридної системи прогнозування енергоспоживання для міських мереж із класичними та сучасними ML-моделями",
    version="0.1.0",
    docs_url="/docs",
    terms_of_service="https://your-domain.com/terms",
    contact={
        "name": "Mykola Kovalenko",
        "url": "https://your-domain.com/contact",
        "email": "kovalenkokola69@gmail.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "docExpansion": "none",
    },
)