from app import create_embeddings
def test_create_embeddings(file='/Users/macbook/Desktop/testPython/cultura_colombiana.pdf'):
    assert create_embeddings(file)

    if file:
        knowledge_base = create_embeddings(file)
        assert knowledge_base
