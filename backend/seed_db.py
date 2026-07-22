from backend.models.database import init_db, get_conn


def seed():
    init_db()
    samples = [
        ("sample_glass.jpg", "Glass", 0.95),
        ("sample_metal.jpg", "Metal", 0.88),
        ("sample_paper.jpg", "Paper", 0.91),
    ]
    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO predictions (filename, top_class, confidence) VALUES (?, ?, ?)",
            samples,
        )
    print("Seeded", len(samples), "predictions.")


if __name__ == "__main__":
    seed()