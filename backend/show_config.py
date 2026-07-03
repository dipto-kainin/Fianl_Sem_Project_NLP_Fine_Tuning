import asyncio
from app.database import AsyncSessionLocal
from app.modules.training.models import TrainingRun
from sqlalchemy import select

async def main():
    # Import models so SQLAlchemy can resolve mapper relationships
    from app.modules.documents.models import Document
    from app.modules.chunks.models import Chunk
    from app.modules.teacher.models import TeacherOutput
    from app.modules.datasets.models import Dataset, DatasetSample
    from app.modules.training.models import TrainingRun
    from app.modules.registry.models import ModelVersion

    async with AsyncSessionLocal() as session:
        res = await session.execute(select(TrainingRun).where(TrainingRun.id == "fd9364e1-23d0-40b8-90d3-24aec1aefb1b"))
        run = res.scalar_one_or_none()
        if run:
            print("Run ID:", run.id)
            print("Status:", run.status)
            print("Training Config:", run.training_config)
            print("Lora Config:", run.lora_config)
            print("Metrics:", run.metrics)
        else:
            print("Run not found")

if __name__ == "__main__":
    asyncio.run(main())
