"""
Unit tests for steps/ingest_data.py

Tests:
    - IngestData class
    - ingest_df ZenML step

Run with:
    pytest tests/test_ingest_data.py -v
"""

import os
import tempfile

import pandas as pd
import pytest

from steps.ingest_data import IngestData, ingest_df


SAMPLE_CSV_CONTENT = """PassengerId,HomePlanet,Age,Transported
0001_01,Earth,39,False
0002_01,Europa,24,True
0003_01,Mars,58,False
"""


@pytest.fixture
def csv_path():
    """
    Her test için geçici bir CSV dosyası oluşturur.

    Test tamamlandıktan sonra dosya ve geçici klasör
    otomatik olarak silinir.
    """

    tmp_dir = tempfile.mkdtemp()
    path = os.path.join(tmp_dir, "sample.csv")

    with open(path, "w", encoding="utf-8") as file:
        file.write(SAMPLE_CSV_CONTENT)

    yield path

    if os.path.exists(path):
        os.remove(path)

    if os.path.exists(tmp_dir):
        os.rmdir(tmp_dir)


@pytest.fixture
def tmp_dir():
    """
    Testler için geçici bir klasör oluşturur.
    """

    directory = tempfile.mkdtemp()

    yield directory

    # Klasör boşsa temizle
    if os.path.exists(directory):
        try:
            os.rmdir(directory)
        except OSError:
            pass


class TestIngestDataClass:

    def test_stores_datapath(self):
        ingestor = IngestData("some/path.csv")

        assert ingestor.datapath == "some/path.csv"

    def test_get_data_returns_dataframe(self, csv_path):
        ingestor = IngestData(csv_path)

        df = ingestor.get_data()

        assert isinstance(df, pd.DataFrame)

    def test_get_data_reads_correct_content(self, csv_path):
        ingestor = IngestData(csv_path)

        df = ingestor.get_data()

        assert len(df) == 3

        assert list(df.columns) == [
            "PassengerId",
            "HomePlanet",
            "Age",
            "Transported",
        ]

        assert df.loc[0, "PassengerId"] == "0001_01"
        assert df.loc[1, "HomePlanet"] == "Europa"
        assert df.loc[2, "Age"] == 58

    def test_get_data_raises_for_missing_file(self, tmp_dir):
        missing_path = os.path.join(
            tmp_dir,
            "does_not_exist.csv",
        )

        ingestor = IngestData(missing_path)

        with pytest.raises(FileNotFoundError):
            ingestor.get_data()

    def test_get_data_raises_for_empty_file(self, tmp_dir):
        empty_path = os.path.join(
            tmp_dir,
            "empty.csv",
        )

        with open(empty_path, "w", encoding="utf-8"):
            pass

        ingestor = IngestData(empty_path)

        with pytest.raises(Exception):
            ingestor.get_data()


class TestIngestDfStep:

    def test_entrypoint_returns_dataframe(self, csv_path):
        df = ingest_df.entrypoint(csv_path)

        assert isinstance(df, pd.DataFrame)

    def test_entrypoint_returns_correct_content(self, csv_path):
        df = ingest_df.entrypoint(csv_path)

        assert len(df) == 3

        assert "Transported" in df.columns

        # NOT: pandas, CSV'deki booleanları Python'un yerleşik `bool` tipi
        # yerine NumPy'nin `np.bool_` tipiyle döndürür. `is True` bir
        # kimlik (identity) karşılaştırmasıdır ve `np.True_ is True` her
        # zaman False verir, çünkü ikisi aynı nesne değildir. Bu yüzden
        # değer eşitliği için `==` kullanıyoruz (ya da bool() ile
        # dönüştürüp `is` kullanabilirsiniz).
        assert df.loc[1, "Transported"] == True  # noqa: E712

    def test_entrypoint_raises_for_missing_file(self, tmp_dir):
        missing_path = os.path.join(
            tmp_dir,
            "nope.csv",
        )

        with pytest.raises(FileNotFoundError):
            ingest_df.entrypoint(missing_path)

    def test_entrypoint_raises_for_invalid_path_type(self):
        """
        data_path None olduğunda pandas tarafından oluşan
        hatanın propagate edildiğini kontrol eder.
        """

        with pytest.raises(Exception):
            ingest_df.entrypoint(None)  # type: ignore[arg-type]