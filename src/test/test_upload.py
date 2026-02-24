"""Tests for /upload/documents and /upload/imu endpoints."""

from fastapi.testclient import TestClient


class TestUploadDocuments:
    """Tests for POST /upload/documents."""

    def test_upload_documents_single_item(self, client: TestClient):
        """Single document item is inserted successfully."""
        payload = {
            "items": [
                {"user": "test_user", "volume": 80, "battery": 85},
            ]
        }
        response = client.post("/upload/documents", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["inserted"] == 1

    def test_upload_documents_multiple_items(self, client: TestClient):
        """Multiple document items are bulk inserted."""
        payload = {
            "items": [
                {"user": "user1", "volume": 80, "gpsLat": 37.77},
                {"user": "user1", "battery": 90, "current_app": "com.example.app"},
                {"user": "user2", "screen_on_ratio": 0.5},
            ]
        }
        response = client.post("/upload/documents", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["inserted"] == 3

    def test_upload_documents_minimal_fields(self, client: TestClient):
        """Only required 'user' field is accepted."""
        payload = {"items": [{"user": "minimal_user"}]}
        response = client.post("/upload/documents", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["inserted"] == 1

    def test_upload_documents_empty_items_rejected(self, client: TestClient):
        """Empty items list returns 422 validation error."""
        payload = {"items": []}
        response = client.post("/upload/documents", json=payload)
        assert response.status_code == 422

    def test_upload_documents_missing_items_rejected(self, client: TestClient):
        """Missing items field returns 422 validation error."""
        payload = {"user": "orphan_user"}
        response = client.post("/upload/documents", json=payload)
        assert response.status_code == 422

    def test_upload_documents_missing_user_rejected(self, client: TestClient):
        """Item without required user field returns 422."""
        payload = {"items": [{"volume": 80}]}
        response = client.post("/upload/documents", json=payload)
        assert response.status_code == 422


class TestUploadImu:
    """Tests for POST /upload/imu."""

    def test_upload_imu_single_item(self, client: TestClient):
        """Single IMU item is inserted successfully."""
        payload = {
            "items": [
                {"user": "test_user", "acc_X": 0.1, "acc_Y": -0.2, "acc_Z": 9.8},
            ]
        }
        response = client.post("/upload/imu", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["inserted"] == 1

    def test_upload_imu_multiple_items(self, client: TestClient):
        """Multiple IMU items are bulk inserted."""
        payload = {
            "items": [
                {"user": "user1", "acc_X": 0.1, "gyro_X": 0.01},
                {"user": "user1", "mag_X": 1.0, "mag_Y": 2.0, "mag_Z": 3.0},
            ]
        }
        response = client.post("/upload/imu", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["inserted"] == 2

    def test_upload_imu_minimal_fields(self, client: TestClient):
        """Only required 'user' field is accepted."""
        payload = {"items": [{"user": "minimal_user"}]}
        response = client.post("/upload/imu", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["inserted"] == 1

    def test_upload_imu_empty_items_rejected(self, client: TestClient):
        """Empty items list returns 422 validation error."""
        payload = {"items": []}
        response = client.post("/upload/imu", json=payload)
        assert response.status_code == 422

    def test_upload_imu_missing_items_rejected(self, client: TestClient):
        """Missing items field returns 422 validation error."""
        payload = {"user": "orphan_user"}
        response = client.post("/upload/imu", json=payload)
        assert response.status_code == 422

    def test_upload_imu_missing_user_rejected(self, client: TestClient):
        """Item without required user field returns 422."""
        payload = {"items": [{"acc_X": 0.1}]}
        response = client.post("/upload/imu", json=payload)
        assert response.status_code == 422
