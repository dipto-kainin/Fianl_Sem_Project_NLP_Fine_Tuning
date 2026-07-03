import asyncio
from app.database import AsyncSessionLocal
from app.modules.datasets.models import Dataset, DatasetSample
from sqlalchemy import select

async def main():
    async with AsyncSessionLocal() as session:
        from app.modules.documents.models import Document
        from app.modules.chunks.models import Chunk
        from app.modules.teacher.models import TeacherOutput
        from app.modules.datasets.models import Dataset, DatasetSample
        from app.modules.training.models import TrainingRun
        from app.modules.registry.models import ModelVersion

        res = await session.execute(select(DatasetSample).limit(10))
        samples = res.scalars().all()
        print("=== Dataset Samples ===")
        for s in samples:
            print(f"Instruction: {s.instruction}\nResponse: {s.response}\n---")

if __name__ == "__main__":
    asyncio.run(main())
