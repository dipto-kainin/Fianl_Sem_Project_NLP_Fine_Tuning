import asyncio
from app.database import AsyncSessionLocal
from app.utils.student_inference import generate_student_answer

async def main():
    # Import models so SQLAlchemy can resolve mapper relationships
    from app.modules.documents.models import Document
    from app.modules.chunks.models import Chunk
    from app.modules.teacher.models import TeacherOutput
    from app.modules.datasets.models import Dataset, DatasetSample
    from app.modules.training.models import TrainingRun
    from app.modules.registry.models import ModelVersion

    async with AsyncSessionLocal() as session:
        # 1. Query email
        print("\nQuery: give me diptodeep's email")
        ans, ver = await generate_student_answer(session, "give me diptodeep's email", [], force_base=False)
        print(f"Answer ({ver}): {ans}")

        # 2. Query work info
        print("\nQuery: where does diptodeep currently work for")
        ans, ver = await generate_student_answer(session, "where does diptodeep currently work for", [], force_base=False)
        print(f"Answer ({ver}): {ans}")

if __name__ == "__main__":
    asyncio.run(main())
