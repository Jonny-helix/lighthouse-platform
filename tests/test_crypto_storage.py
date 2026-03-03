"""
Tests for the crypto and storage modules of LIGHTHOUSE.

Covers:
  - Crypto constants (MAGIC, HEADER_SIZE)
  - Key derivation (Argon2id)
  - Encrypt / decrypt roundtrip (low-level and file-level)
  - File format structure (header, magic bytes)
  - Serialize / deserialize KB roundtrip
  - save_kb / load_kb roundtrip with encrypted .lighthouse files
  - Wrong-passphrase rejection
  - KB data integrity through roundtrip (facts, sources)
  - create_kb produces valid WorkingLayer
  - export_kb_json / load_kb_json unencrypted roundtrip
"""

import sys
import os
import json
import tempfile
import pytest
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lighthouse.crypto import (
    MAGIC, HEADER_SIZE, VERSION,
    KEY_TYPE_DATA, KEY_TYPE_REFERENCE,
    derive_key,
    encrypt, decrypt,
    encrypt_to_file, decrypt_from_file,
    save_encrypted, load_encrypted,
    pack_encrypted_file, unpack_encrypted_file,
)
from lighthouse.storage import (
    serialize_kb, deserialize_kb,
    save_kb, load_kb,
    export_kb_json, load_kb_json,
)
from lighthouse.schema import create_kb, Fact, Source, WorkingLayer


# ============ HELPERS ============

PASSPHRASE = "test-passphrase-42!"
WRONG_PASSPHRASE = "wrong-passphrase-99!"


def _make_kb_with_data():
    """Create a KB seeded with a fact and a source for roundtrip tests."""
    kb = create_kb(name="Test Practice", domain="coaching")
    fact = Fact(
        fact_id="f001",
        fact_type="finding",
        statement="Coaching improves employee retention by 20%.",
        source_refs=["s001"],
        category="Effectiveness",
        evidence_level="III",
    )
    source = Source(
        source_id="s001",
        title="Coaching Outcomes Meta-Analysis",
        authors="Smith, J. and Doe, A.",
        publication_year=2024,
        journal="Journal of Coaching Research",
    )
    kb.facts.append(fact)
    kb.sources.append(source)
    return kb


# ============ CRYPTO CONSTANTS ============


class TestCryptoConstants:
    """Verify fundamental crypto constants match the LIGHTHOUSE spec."""

    def test_magic_bytes_are_lths(self):
        """MAGIC must be b'LTHS', not b'BNYN' (BANYAN)."""
        assert MAGIC == b'LTHS'

    def test_magic_is_not_banyan(self):
        """Guard against accidental BANYAN magic leaking in."""
        assert MAGIC != b'BNYN'

    def test_header_size_is_50(self):
        """Header is exactly 50 bytes per the file format spec."""
        assert HEADER_SIZE == 50

    def test_version_is_1(self):
        assert VERSION == 1


# ============ KEY DERIVATION ============


class TestKeyDerivation:
    """Tests for Argon2id key derivation."""

    def test_derive_key_produces_32_byte_key(self):
        key, salt = derive_key(PASSPHRASE)
        assert len(key) == 32

    def test_derive_key_produces_16_byte_salt(self):
        key, salt = derive_key(PASSPHRASE)
        assert len(salt) == 16

    def test_derive_key_deterministic_with_same_salt(self):
        """Same passphrase + same salt must yield the same key."""
        key1, salt = derive_key(PASSPHRASE)
        key2, _ = derive_key(PASSPHRASE, salt)
        assert key1 == key2

    def test_derive_key_different_salt_yields_different_key(self):
        key1, salt1 = derive_key(PASSPHRASE)
        key2, salt2 = derive_key(PASSPHRASE)
        # Two random salts should differ (astronomically unlikely to collide)
        if salt1 != salt2:
            assert key1 != key2


# ============ ENCRYPT / DECRYPT (LOW-LEVEL) ============


class TestEncryptDecrypt:
    """Tests for AES-256-GCM encrypt/decrypt at the byte level."""

    def test_roundtrip_bytes(self):
        """encrypt then decrypt must recover the original plaintext."""
        plaintext = b"Hello, LIGHTHOUSE!"
        key, _ = derive_key(PASSPHRASE)

        ciphertext, iv, tag = encrypt(plaintext, key)
        recovered = decrypt(ciphertext, key, iv, tag)
        assert recovered == plaintext

    def test_wrong_key_fails(self):
        """Decrypting with a different key must raise an error."""
        plaintext = b"secret data"
        key_good, _ = derive_key(PASSPHRASE)
        key_bad, _ = derive_key(WRONG_PASSPHRASE)

        ciphertext, iv, tag = encrypt(plaintext, key_good)

        with pytest.raises(Exception):
            decrypt(ciphertext, key_bad, iv, tag)

    def test_ciphertext_differs_from_plaintext(self):
        plaintext = b"This is sensitive"
        key, _ = derive_key(PASSPHRASE)
        ciphertext, iv, tag = encrypt(plaintext, key)
        assert ciphertext != plaintext

    def test_empty_plaintext_roundtrip(self):
        """Edge case: encrypting empty bytes should still roundtrip."""
        key, _ = derive_key(PASSPHRASE)
        ciphertext, iv, tag = encrypt(b"", key)
        recovered = decrypt(ciphertext, key, iv, tag)
        assert recovered == b""


# ============ FILE FORMAT (encrypt_to_file / decrypt_from_file) ============


class TestFileFormat:
    """Tests for the LIGHTHOUSE encrypted file format packing."""

    def test_encrypt_to_file_starts_with_magic(self):
        """Packed file data must start with LTHS magic bytes."""
        data = b"payload"
        file_bytes = encrypt_to_file(data, PASSPHRASE)
        assert file_bytes[:4] == b'LTHS'

    def test_encrypt_to_file_header_is_50_bytes(self):
        """The header portion is exactly 50 bytes; ciphertext follows."""
        data = b"payload"
        file_bytes = encrypt_to_file(data, PASSPHRASE)
        # Total must be at least header + some ciphertext
        assert len(file_bytes) >= HEADER_SIZE

    def test_encrypt_decrypt_file_roundtrip(self):
        """encrypt_to_file / decrypt_from_file must roundtrip."""
        data = b'{"name": "Test KB"}'
        file_bytes = encrypt_to_file(data, PASSPHRASE)
        recovered, key_type = decrypt_from_file(file_bytes, PASSPHRASE)
        assert recovered == data
        assert key_type == KEY_TYPE_DATA

    def test_decrypt_wrong_passphrase_raises(self):
        data = b"secret"
        file_bytes = encrypt_to_file(data, PASSPHRASE)
        with pytest.raises(Exception):
            decrypt_from_file(file_bytes, WRONG_PASSPHRASE)

    def test_key_type_reference_preserved(self):
        """Key type should survive the pack/unpack cycle."""
        data = b"archived content"
        file_bytes = encrypt_to_file(data, PASSPHRASE, key_type=KEY_TYPE_REFERENCE)
        _, key_type = decrypt_from_file(file_bytes, PASSPHRASE)
        assert key_type == KEY_TYPE_REFERENCE

    def test_save_load_encrypted_temp_file(self):
        """save_encrypted / load_encrypted roundtrip via a real temp file."""
        data = b"disk-level test"
        with tempfile.NamedTemporaryFile(suffix=".lighthouse", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            save_encrypted(tmp_path, data, PASSPHRASE)
            recovered, key_type = load_encrypted(tmp_path, PASSPHRASE)
            assert recovered == data
            assert key_type == KEY_TYPE_DATA
        finally:
            os.unlink(tmp_path)


# ============ SERIALIZE / DESERIALIZE KB ============


class TestSerializeDeserialize:
    """Tests for WorkingLayer serialization to/from JSON bytes."""

    def test_roundtrip_empty_kb(self):
        kb = create_kb(name="Empty", domain="coaching")
        data = serialize_kb(kb)
        recovered = deserialize_kb(data)
        assert recovered.metadata.name == "Empty"
        assert recovered.metadata.domain == "coaching"

    def test_roundtrip_with_facts_and_sources(self):
        kb = _make_kb_with_data()
        data = serialize_kb(kb)
        recovered = deserialize_kb(data)
        assert len(recovered.facts) == 1
        assert recovered.facts[0].fact_id == "f001"
        assert recovered.facts[0].statement == "Coaching improves employee retention by 20%."
        assert len(recovered.sources) == 1
        assert recovered.sources[0].source_id == "s001"

    def test_serialized_data_is_valid_json(self):
        kb = create_kb(name="JSON Check", domain="coaching")
        data = serialize_kb(kb)
        parsed = json.loads(data.decode("utf-8"))
        assert parsed["metadata"]["name"] == "JSON Check"


# ============ save_kb / load_kb (FULL ENCRYPTED ROUNDTRIP) ============


class TestSaveLoadKB:
    """Tests for the high-level save_kb / load_kb with encryption."""

    def test_save_load_roundtrip(self):
        kb = _make_kb_with_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test")
            info = save_kb(kb, filepath, PASSPHRASE)
            # save_kb appends .lighthouse
            assert info["filepath"].endswith(".lighthouse")

            loaded_kb, load_info = load_kb(info["filepath"], PASSPHRASE)
            assert loaded_kb.metadata.name == "Test Practice"
            assert len(loaded_kb.facts) == 1
            assert len(loaded_kb.sources) == 1

    def test_save_kb_creates_lighthouse_extension(self):
        kb = create_kb(name="Ext Test", domain="coaching")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "myfile")
            info = save_kb(kb, filepath, PASSPHRASE)
            assert info["filepath"].endswith(".lighthouse")
            assert os.path.exists(info["filepath"])

    def test_save_kb_already_has_extension(self):
        """If the path already has .lighthouse, it should NOT double up."""
        kb = create_kb(name="Double Ext", domain="coaching")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "data.lighthouse")
            info = save_kb(kb, filepath, PASSPHRASE)
            assert info["filepath"].endswith(".lighthouse")
            assert not info["filepath"].endswith(".lighthouse.lighthouse")

    def test_load_wrong_passphrase_raises(self):
        kb = create_kb(name="Locked", domain="coaching")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "locked")
            save_kb(kb, filepath, PASSPHRASE)
            actual_path = filepath + ".lighthouse"
            with pytest.raises(Exception):
                load_kb(actual_path, WRONG_PASSPHRASE)

    def test_kb_data_survives_roundtrip(self):
        """Verify fact and source fields survive encrypt/decrypt cycle."""
        kb = _make_kb_with_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "integrity")
            info = save_kb(kb, filepath, PASSPHRASE)
            loaded_kb, _ = load_kb(info["filepath"], PASSPHRASE)

            f = loaded_kb.facts[0]
            assert f.fact_id == "f001"
            assert f.fact_type == "finding"
            assert f.statement == "Coaching improves employee retention by 20%."
            assert f.source_refs == ["s001"]
            assert f.category == "Effectiveness"

            s = loaded_kb.sources[0]
            assert s.source_id == "s001"
            assert s.title == "Coaching Outcomes Meta-Analysis"
            assert s.authors == "Smith, J. and Doe, A."
            assert s.publication_year == 2024
            assert s.journal == "Journal of Coaching Research"

    def test_file_starts_with_lths_magic(self):
        """The saved .lighthouse file on disk must start with LTHS magic."""
        kb = create_kb(name="Magic Check", domain="coaching")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "magic")
            info = save_kb(kb, filepath, PASSPHRASE)
            with open(info["filepath"], "rb") as f:
                header = f.read(4)
            assert header == b'LTHS'

    def test_save_info_contains_expected_keys(self):
        kb = create_kb(name="Info Test", domain="coaching")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "info")
            info = save_kb(kb, filepath, PASSPHRASE)
            assert "filepath" in info
            assert "size_bytes" in info
            assert "encrypted" in info
            assert info["encrypted"] is True


# ============ create_kb ============


class TestCreateKB:
    """Tests for the create_kb factory function."""

    def test_creates_working_layer(self):
        kb = create_kb(name="My Practice", domain="coaching")
        assert isinstance(kb, WorkingLayer)

    def test_metadata_domain_is_coaching(self):
        kb = create_kb(name="Domain Test", domain="coaching")
        assert kb.metadata.domain == "coaching"

    def test_default_domain_is_coaching(self):
        """If no domain given, default should be 'coaching'."""
        kb = create_kb(name="Default Domain")
        assert kb.metadata.domain == "coaching"

    def test_empty_facts_and_sources(self):
        kb = create_kb(name="Fresh KB", domain="coaching")
        assert kb.facts == []
        assert kb.sources == []

    def test_audit_log_has_create_entry(self):
        kb = create_kb(name="Audit Test", domain="coaching")
        assert len(kb.audit_log) >= 1
        last = kb.audit_log[-1]
        assert last.action == "create"


# ============ export_kb_json / load_kb_json (UNENCRYPTED) ============


class TestJsonExportLoad:
    """Tests for unencrypted JSON export and load (demo mode)."""

    def test_export_load_roundtrip(self):
        kb = _make_kb_with_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "demo")
            info = export_kb_json(kb, filepath)
            assert info["filepath"].endswith(".json")
            assert info["encrypted"] is False

            loaded_kb, load_info = load_kb_json(info["filepath"])
            assert loaded_kb.metadata.name == "Test Practice"
            assert len(loaded_kb.facts) == 1
            assert loaded_kb.facts[0].fact_id == "f001"

    def test_export_creates_valid_json_file(self):
        kb = create_kb(name="JSON Valid", domain="coaching")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "valid")
            info = export_kb_json(kb, filepath)
            with open(info["filepath"], "r", encoding="utf-8") as f:
                parsed = json.load(f)
            assert parsed["metadata"]["name"] == "JSON Valid"

    def test_export_appends_json_extension(self):
        kb = create_kb(name="Ext JSON", domain="coaching")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "noext")
            info = export_kb_json(kb, filepath)
            assert info["filepath"].endswith(".json")
