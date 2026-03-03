"""
LIGHTHOUSE Encryption Module

Implements the triple-key encryption model:
- Customer Data Key: Customer-controlled, permanent access to their content
- Reference Key: High-friction access to archived/sensitive content
- Platform Key: Subscription-gated access to platform features

Uses Argon2id for key derivation and AES-256-GCM for encryption.

v3 format adds multi-user PIN encryption with a random data key wrapped
per-user (PBKDF2-SHA256 for PINs, Argon2id for master code).
"""

import os
import json
import struct
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from argon2.low_level import hash_secret_raw, Type
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


# ============ CONSTANTS ============

# File format magic number: "LTHS" in bytes
MAGIC = b'LTHS'
VERSION = 1

# Key types
KEY_TYPE_DATA = 1       # Customer Data Key
KEY_TYPE_REFERENCE = 2  # Reference Key (archive)
KEY_TYPE_PLATFORM = 3   # Platform Key (subscription)

# Argon2id parameters (secure defaults)
ARGON2_TIME_COST = 3        # iterations
ARGON2_MEMORY_COST = 65536  # 64 MB
ARGON2_PARALLELISM = 4      # threads
ARGON2_HASH_LEN = 32        # 256 bits
ARGON2_SALT_LEN = 16        # 128 bits

# AES-GCM parameters
AES_KEY_LEN = 32    # 256 bits
AES_IV_LEN = 12     # 96 bits (standard for GCM)
AES_TAG_LEN = 16    # 128 bits

# v3 multi-user format
V3_FORMAT_VERSION = 3
V3_SEPARATOR = b'\n---PAYLOAD---\n'
PBKDF2_PIN_ITERATIONS = 600_000  # 10K PINs * 600K = ~6B ops to brute-force
PBKDF2_SALT_LEN = 16             # 128 bits
DATA_KEY_LEN = 32                # 256 bits
# Wrapped data key: nonce(12) + tag(16) + ciphertext(32) = 60 bytes
WRAPPED_KEY_LEN = AES_IV_LEN + AES_TAG_LEN + DATA_KEY_LEN


# ============ KEY DERIVATION ============

def derive_key(passphrase: str, salt: bytes = None) -> Tuple[bytes, bytes]:
    """
    Derive a 256-bit encryption key from a passphrase using Argon2id.

    Args:
        passphrase: User-provided passphrase
        salt: Optional salt (generated if not provided)

    Returns:
        Tuple of (key, salt)
    """
    if salt is None:
        salt = os.urandom(ARGON2_SALT_LEN)

    if isinstance(passphrase, str):
        passphrase = passphrase.encode('utf-8')

    key = hash_secret_raw(
        secret=passphrase,
        salt=salt,
        time_cost=ARGON2_TIME_COST,
        memory_cost=ARGON2_MEMORY_COST,
        parallelism=ARGON2_PARALLELISM,
        hash_len=ARGON2_HASH_LEN,
        type=Type.ID  # Argon2id
    )

    return key, salt


# ============ ENCRYPTION / DECRYPTION ============

def encrypt(plaintext: bytes, key: bytes) -> Tuple[bytes, bytes, bytes]:
    """
    Encrypt data using AES-256-GCM.

    Args:
        plaintext: Data to encrypt
        key: 256-bit encryption key

    Returns:
        Tuple of (ciphertext, iv, tag)
    """
    iv = os.urandom(AES_IV_LEN)
    aesgcm = AESGCM(key)

    # AESGCM.encrypt returns ciphertext + tag concatenated
    ciphertext_with_tag = aesgcm.encrypt(iv, plaintext, None)

    # Split into ciphertext and tag
    ciphertext = ciphertext_with_tag[:-AES_TAG_LEN]
    tag = ciphertext_with_tag[-AES_TAG_LEN:]

    return ciphertext, iv, tag


def decrypt(ciphertext: bytes, key: bytes, iv: bytes, tag: bytes) -> bytes:
    """
    Decrypt data using AES-256-GCM.

    Args:
        ciphertext: Encrypted data
        key: 256-bit encryption key
        iv: Initialization vector
        tag: Authentication tag

    Returns:
        Decrypted plaintext

    Raises:
        cryptography.exceptions.InvalidTag: If authentication fails
    """
    aesgcm = AESGCM(key)

    # AESGCM.decrypt expects ciphertext + tag concatenated
    ciphertext_with_tag = ciphertext + tag

    plaintext = aesgcm.decrypt(iv, ciphertext_with_tag, None)
    return plaintext


# ============ FILE FORMAT ============

"""
LIGHTHOUSE Encrypted File Format:

Offset  Size    Field
------  ----    -----
0       4       Magic ("LTHS")
4       1       Version (1)
5       1       Key Type (1=Data, 2=Reference, 3=Platform)
6       16      Salt (Argon2)
22      12      IV (AES-GCM)
34      16      Tag (AES-GCM)
50      ...     Ciphertext

Total header: 50 bytes
"""

HEADER_SIZE = 50


def pack_encrypted_file(ciphertext: bytes, salt: bytes, iv: bytes,
                        tag: bytes, key_type: int = KEY_TYPE_DATA) -> bytes:
    """
    Pack encrypted data into LIGHTHOUSE file format.

    Args:
        ciphertext: Encrypted data
        salt: Argon2 salt
        iv: AES-GCM IV
        tag: AES-GCM authentication tag
        key_type: Type of key used (1=Data, 2=Reference, 3=Platform)

    Returns:
        Complete encrypted file as bytes
    """
    # Validate lengths
    assert len(salt) == ARGON2_SALT_LEN, f"Salt must be {ARGON2_SALT_LEN} bytes"
    assert len(iv) == AES_IV_LEN, f"IV must be {AES_IV_LEN} bytes"
    assert len(tag) == AES_TAG_LEN, f"Tag must be {AES_TAG_LEN} bytes"

    # Pack header
    header = struct.pack(
        f'4sBB{ARGON2_SALT_LEN}s{AES_IV_LEN}s{AES_TAG_LEN}s',
        MAGIC,
        VERSION,
        key_type,
        salt,
        iv,
        tag
    )

    return header + ciphertext


def unpack_encrypted_file(data: bytes) -> Tuple[bytes, bytes, bytes, bytes, int]:
    """
    Unpack LIGHTHOUSE encrypted file format.

    Args:
        data: Complete encrypted file as bytes

    Returns:
        Tuple of (ciphertext, salt, iv, tag, key_type)

    Raises:
        ValueError: If file format is invalid
    """
    if len(data) < HEADER_SIZE:
        raise ValueError("File too small to be a valid LIGHTHOUSE file")

    # Unpack header
    header_format = f'4sBB{ARGON2_SALT_LEN}s{AES_IV_LEN}s{AES_TAG_LEN}s'
    magic, version, key_type, salt, iv, tag = struct.unpack(
        header_format,
        data[:HEADER_SIZE]
    )

    # Validate magic
    if magic != MAGIC:
        raise ValueError("Invalid file format: not a LIGHTHOUSE file")

    # Validate version
    if version != VERSION:
        raise ValueError(f"Unsupported file version: {version}")

    # Extract ciphertext
    ciphertext = data[HEADER_SIZE:]

    return ciphertext, salt, iv, tag, key_type


# ============ HIGH-LEVEL API ============

def encrypt_to_file(data: bytes, passphrase: str,
                    key_type: int = KEY_TYPE_DATA) -> bytes:
    """
    Encrypt data and pack into LIGHTHOUSE file format.

    Args:
        data: Data to encrypt (typically JSON)
        passphrase: User passphrase
        key_type: Type of key (1=Data, 2=Reference, 3=Platform)

    Returns:
        Complete encrypted file as bytes
    """
    # Derive key
    key, salt = derive_key(passphrase)

    # Encrypt
    ciphertext, iv, tag = encrypt(data, key)

    # Pack
    return pack_encrypted_file(ciphertext, salt, iv, tag, key_type)


def decrypt_from_file(file_data: bytes, passphrase: str) -> Tuple[bytes, int]:
    """
    Decrypt LIGHTHOUSE file format.

    Args:
        file_data: Complete encrypted file as bytes
        passphrase: User passphrase

    Returns:
        Tuple of (decrypted data, key_type)

    Raises:
        ValueError: If file format is invalid
        cryptography.exceptions.InvalidTag: If passphrase is wrong
    """
    # Unpack
    ciphertext, salt, iv, tag, key_type = unpack_encrypted_file(file_data)

    # Derive key with stored salt
    key, _ = derive_key(passphrase, salt)

    # Decrypt
    plaintext = decrypt(ciphertext, key, iv, tag)

    return plaintext, key_type


def save_encrypted(filepath: str, data: bytes, passphrase: str,
                   key_type: int = KEY_TYPE_DATA):
    """
    Encrypt data and save to file.

    Args:
        filepath: Path to save to
        data: Data to encrypt
        passphrase: User passphrase
        key_type: Type of key
    """
    encrypted = encrypt_to_file(data, passphrase, key_type)
    with open(filepath, 'wb') as f:
        f.write(encrypted)


def load_encrypted(filepath: str, passphrase: str) -> Tuple[bytes, int]:
    """
    Load and decrypt file.

    Args:
        filepath: Path to load from
        passphrase: User passphrase

    Returns:
        Tuple of (decrypted data, key_type)
    """
    with open(filepath, 'rb') as f:
        file_data = f.read()

    return decrypt_from_file(file_data, passphrase)


# ============ JSON HELPERS ============

def encrypt_json(obj: dict, passphrase: str,
                 key_type: int = KEY_TYPE_DATA) -> bytes:
    """
    Serialize dict to JSON and encrypt.

    Args:
        obj: Dictionary to encrypt
        passphrase: User passphrase
        key_type: Type of key

    Returns:
        Encrypted file as bytes
    """
    json_bytes = json.dumps(obj, default=str, ensure_ascii=False).encode('utf-8')
    return encrypt_to_file(json_bytes, passphrase, key_type)


def decrypt_json(file_data: bytes, passphrase: str) -> Tuple[dict, int]:
    """
    Decrypt file and parse JSON.

    Args:
        file_data: Encrypted file as bytes
        passphrase: User passphrase

    Returns:
        Tuple of (parsed dict, key_type)
    """
    plaintext, key_type = decrypt_from_file(file_data, passphrase)
    obj = json.loads(plaintext.decode('utf-8'))
    return obj, key_type


def save_json_encrypted(filepath: str, obj: dict, passphrase: str,
                        key_type: int = KEY_TYPE_DATA):
    """
    Serialize dict to JSON, encrypt, and save.
    """
    encrypted = encrypt_json(obj, passphrase, key_type)
    with open(filepath, 'wb') as f:
        f.write(encrypted)


def load_json_encrypted(filepath: str, passphrase: str) -> Tuple[dict, int]:
    """
    Load file, decrypt, and parse JSON.
    """
    with open(filepath, 'rb') as f:
        file_data = f.read()
    return decrypt_json(file_data, passphrase)


# ============ VERIFICATION ============

def verify_file(filepath: str) -> dict:
    """
    Verify a LIGHTHOUSE file without decrypting.

    Args:
        filepath: Path to file

    Returns:
        Dict with file info (valid, version, key_type, size)
    """
    try:
        with open(filepath, 'rb') as f:
            data = f.read(HEADER_SIZE + 100)  # Read header + some ciphertext

        if len(data) < HEADER_SIZE:
            return {"valid": False, "error": "File too small"}

        magic = data[:4]
        if magic != MAGIC:
            return {"valid": False, "error": "Not a LIGHTHOUSE file"}

        version = data[4]
        key_type = data[5]

        # Get full file size
        with open(filepath, 'rb') as f:
            f.seek(0, 2)  # Seek to end
            total_size = f.tell()

        key_type_names = {
            KEY_TYPE_DATA: "Customer Data Key",
            KEY_TYPE_REFERENCE: "Reference Key",
            KEY_TYPE_PLATFORM: "Platform Key"
        }

        return {
            "valid": True,
            "version": version,
            "key_type": key_type,
            "key_type_name": key_type_names.get(key_type, "Unknown"),
            "total_size": total_size,
            "ciphertext_size": total_size - HEADER_SIZE
        }

    except Exception as e:
        return {"valid": False, "error": str(e)}


# ============ V3 MULTI-USER PIN ENCRYPTION ============

"""
LIGHTHOUSE v3 Encrypted File Format:

[4 bytes : uint32 BE header_length]
[header_length bytes : UTF-8 JSON header]   <- unencrypted, contains user slots
[15 bytes : b'\\n---PAYLOAD---\\n']          <- separator
[12 bytes : payload nonce]
[16 bytes : payload auth tag]
[remaining : AES-256-GCM ciphertext]

The JSON header contains per-user encrypted copies of the random data key,
allowing any authorised user (or the master code) to decrypt the payload.
"""


def derive_pin_key(pin: str, salt: bytes = None) -> Tuple[bytes, bytes]:
    """Derive a 256-bit key from a 4-digit PIN using PBKDF2-HMAC-SHA256.

    PBKDF2 with 600K iterations makes brute-forcing 10K possible PINs
    computationally expensive (~hours).

    Args:
        pin: 4-digit PIN string (e.g. "1234")
        salt: Optional salt (generated if not provided)

    Returns:
        Tuple of (key: bytes, salt: bytes)
    """
    if salt is None:
        salt = os.urandom(PBKDF2_SALT_LEN)

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=DATA_KEY_LEN,
        salt=salt,
        iterations=PBKDF2_PIN_ITERATIONS,
    )
    key = kdf.derive(pin.encode("utf-8"))
    return key, salt


def encrypt_data_key(data_key: bytes, wrapping_key: bytes) -> bytes:
    """Encrypt a 32-byte data key with a wrapping key (AES-256-GCM).

    Returns:
        60 bytes: nonce(12) + tag(16) + ciphertext(32)
    """
    nonce = os.urandom(AES_IV_LEN)
    aesgcm = AESGCM(wrapping_key)
    ct_with_tag = aesgcm.encrypt(nonce, data_key, None)
    # AESGCM returns ct + tag concatenated
    ct = ct_with_tag[:-AES_TAG_LEN]
    tag = ct_with_tag[-AES_TAG_LEN:]
    return nonce + tag + ct


def decrypt_data_key(blob: bytes, wrapping_key: bytes) -> bytes:
    """Decrypt a wrapped data key (60 bytes -> 32-byte data key).

    Args:
        blob: 60 bytes = nonce(12) + tag(16) + ciphertext(32)
        wrapping_key: 32-byte AES key

    Returns:
        32-byte data key

    Raises:
        cryptography.exceptions.InvalidTag: if wrapping key is wrong
    """
    if len(blob) != WRAPPED_KEY_LEN:
        raise ValueError(
            f"Wrapped key must be {WRAPPED_KEY_LEN} bytes, got {len(blob)}"
        )
    nonce = blob[:AES_IV_LEN]
    tag = blob[AES_IV_LEN : AES_IV_LEN + AES_TAG_LEN]
    ct = blob[AES_IV_LEN + AES_TAG_LEN :]
    aesgcm = AESGCM(wrapping_key)
    return aesgcm.decrypt(nonce, ct + tag, None)


def detect_format(data: bytes) -> int:
    """Detect the LIGHTHOUSE file format version from raw file bytes.

    Returns:
        1 for v1 (LTHS magic), 3 for v3 (JSON header), 0 for unknown
    """
    if len(data) < 4:
        return 0

    # v1: starts with b'LTHS'
    if data[:4] == MAGIC:
        return 1

    # v3: starts with a 4-byte big-endian header length followed by JSON
    try:
        header_len = struct.unpack(">I", data[:4])[0]
        if 10 < header_len < 1_000_000 and len(data) >= 4 + header_len:
            header_json = data[4 : 4 + header_len]
            header = json.loads(header_json.decode("utf-8"))
            if header.get("format_version") == V3_FORMAT_VERSION:
                return 3
    except Exception:
        pass

    return 0


def pack_v3_file(
    header_dict: dict,
    payload_bytes: bytes,
    user_pins: Dict[str, dict],
    master_code: Optional[str] = None,
) -> bytes:
    """Assemble a v3 encrypted file.

    Args:
        header_dict: Base header fields (display_name, project_code, etc.).
            Must NOT contain 'users' or 'master' -- those are built here.
        payload_bytes: Serialised KB JSON bytes (plaintext).
        user_pins: Mapping of user_id -> {"pin": str, "user_name": str}.
            At least one user is required.
        master_code: Optional master unlock code (8+ chars, hashed via Argon2id).

    Returns:
        Complete v3 file as bytes.
    """
    if not user_pins:
        raise ValueError("At least one user PIN is required")

    # 1. Generate random data key
    data_key = os.urandom(DATA_KEY_LEN)

    # 2. Encrypt payload with data key
    payload_nonce = os.urandom(AES_IV_LEN)
    aesgcm = AESGCM(data_key)
    ct_with_tag = aesgcm.encrypt(payload_nonce, payload_bytes, None)
    payload_ct = ct_with_tag[:-AES_TAG_LEN]
    payload_tag = ct_with_tag[-AES_TAG_LEN:]

    # 3. Wrap data key for each user (PIN -> PBKDF2 -> wrapping key)
    users_list = []
    now_iso = datetime.now().strftime("%Y-%m-%d")
    for user_id, info in user_pins.items():
        pin = info["pin"]
        user_name = info.get("user_name", "")
        pin_key, pin_salt = derive_pin_key(pin)
        wrapped = encrypt_data_key(data_key, pin_key)
        users_list.append({
            "user_id": user_id,
            "user_name": user_name,
            "pin_salt": pin_salt.hex(),
            "encrypted_data_key": wrapped.hex(),
            "created": now_iso,
            "last_accessed": now_iso,
        })

    # 4. Wrap data key for master code (Argon2id -> wrapping key)
    master_block = None
    if master_code:
        master_key, master_salt = derive_key(master_code)
        wrapped_master = encrypt_data_key(data_key, master_key)
        master_block = {
            "salt": master_salt.hex(),
            "encrypted_data_key": wrapped_master.hex(),
        }

    # 5. Build complete header
    header = dict(header_dict)
    header["format_version"] = V3_FORMAT_VERSION
    header["users"] = users_list
    if master_block:
        header["master"] = master_block

    header_json = json.dumps(header, ensure_ascii=False).encode("utf-8")
    header_len = struct.pack(">I", len(header_json))

    # 6. Assemble file
    return (
        header_len
        + header_json
        + V3_SEPARATOR
        + payload_nonce
        + payload_tag
        + payload_ct
    )


def unpack_v3_header(data: bytes) -> dict:
    """Read the JSON header from a v3 file WITHOUT decrypting the payload.

    Returns:
        Parsed header dict (contains users list, master block, stats, etc.)

    Raises:
        ValueError: if data is not a valid v3 file
    """
    if len(data) < 8:
        raise ValueError("File too small for v3 format")

    header_len = struct.unpack(">I", data[:4])[0]
    if header_len < 10 or header_len > 10_000_000:
        raise ValueError(f"Invalid v3 header length: {header_len}")

    if len(data) < 4 + header_len:
        raise ValueError("File truncated: header incomplete")

    header_json = data[4 : 4 + header_len]
    header = json.loads(header_json.decode("utf-8"))

    if header.get("format_version") != V3_FORMAT_VERSION:
        raise ValueError(
            f"Expected format_version {V3_FORMAT_VERSION}, "
            f"got {header.get('format_version')}"
        )

    return header


def unpack_v3_payload(data: bytes, data_key: bytes) -> bytes:
    """Extract and decrypt the payload from a v3 file.

    Args:
        data: Complete v3 file bytes.
        data_key: 32-byte data key (already unwrapped from a user slot or master).

    Returns:
        Decrypted payload bytes (KB JSON).

    Raises:
        ValueError: if separator not found or structure invalid
        cryptography.exceptions.InvalidTag: if data_key is wrong
    """
    # Find separator
    sep_idx = data.find(V3_SEPARATOR)
    if sep_idx == -1:
        raise ValueError("v3 payload separator not found")

    payload_start = sep_idx + len(V3_SEPARATOR)
    remaining = data[payload_start:]

    if len(remaining) < AES_IV_LEN + AES_TAG_LEN + 1:
        raise ValueError("v3 payload too short")

    nonce = remaining[:AES_IV_LEN]
    tag = remaining[AES_IV_LEN : AES_IV_LEN + AES_TAG_LEN]
    ct = remaining[AES_IV_LEN + AES_TAG_LEN :]

    aesgcm = AESGCM(data_key)
    return aesgcm.decrypt(nonce, ct + tag, None)


def v3_decrypt_with_pin(
    data: bytes, user_id: str, pin: str
) -> Tuple[bytes, dict]:
    """Convenience: decrypt a v3 file using a user's PIN.

    Args:
        data: Complete v3 file bytes.
        user_id: The user_id to look up in the header.
        pin: The user's 4-digit PIN.

    Returns:
        Tuple of (decrypted payload bytes, header dict)

    Raises:
        ValueError: if user not found
        cryptography.exceptions.InvalidTag: if PIN is wrong
    """
    header = unpack_v3_header(data)

    # Find user slot
    user_slot = None
    for u in header.get("users", []):
        if u["user_id"] == user_id:
            user_slot = u
            break
    if user_slot is None:
        raise ValueError(f"User '{user_id}' not found in v3 header")

    # Derive PIN key and unwrap data key
    pin_salt = bytes.fromhex(user_slot["pin_salt"])
    pin_key, _ = derive_pin_key(pin, pin_salt)
    wrapped = bytes.fromhex(user_slot["encrypted_data_key"])
    data_key = decrypt_data_key(wrapped, pin_key)

    # Decrypt payload
    payload = unpack_v3_payload(data, data_key)
    return payload, header


def v3_decrypt_with_master(data: bytes, master_code: str) -> Tuple[bytes, dict]:
    """Convenience: decrypt a v3 file using the master code.

    Args:
        data: Complete v3 file bytes.
        master_code: Master unlock code.

    Returns:
        Tuple of (decrypted payload bytes, header dict)

    Raises:
        ValueError: if no master block in header
        cryptography.exceptions.InvalidTag: if master code is wrong
    """
    header = unpack_v3_header(data)

    master_block = header.get("master")
    if not master_block:
        raise ValueError("No master unlock configured for this file")

    master_salt = bytes.fromhex(master_block["salt"])
    master_key, _ = derive_key(master_code, master_salt)
    wrapped = bytes.fromhex(master_block["encrypted_data_key"])
    data_key = decrypt_data_key(wrapped, master_key)

    payload = unpack_v3_payload(data, data_key)
    return payload, header


def v3_add_user(
    data: bytes,
    new_user_id: str,
    new_user_name: str,
    new_pin: str,
    data_key: bytes,
) -> bytes:
    """Add a new user to an existing v3 file.

    Requires the already-decrypted data_key (caller must authenticate first).

    Args:
        data: Complete v3 file bytes.
        new_user_id: New user identifier.
        new_user_name: Display name for the new user.
        new_pin: New user's 4-digit PIN.
        data_key: The 32-byte data key (unwrapped by caller).

    Returns:
        Updated v3 file bytes with the new user added.
    """
    header = unpack_v3_header(data)

    # Check for duplicate
    for u in header.get("users", []):
        if u["user_id"] == new_user_id:
            raise ValueError(f"User '{new_user_id}' already exists")

    # Wrap data key for new user
    pin_key, pin_salt = derive_pin_key(new_pin)
    wrapped = encrypt_data_key(data_key, pin_key)
    now_iso = datetime.now().strftime("%Y-%m-%d")

    header["users"].append({
        "user_id": new_user_id,
        "user_name": new_user_name,
        "pin_salt": pin_salt.hex(),
        "encrypted_data_key": wrapped.hex(),
        "created": now_iso,
        "last_accessed": now_iso,
    })

    # Rebuild file with updated header, same payload bytes
    sep_idx = data.find(V3_SEPARATOR)
    payload_section = data[sep_idx:]  # separator + nonce + tag + ct

    header_json = json.dumps(header, ensure_ascii=False).encode("utf-8")
    header_len = struct.pack(">I", len(header_json))
    return header_len + header_json + payload_section


def v3_remove_user(data: bytes, target_user_id: str) -> bytes:
    """Remove a user from an existing v3 file.

    Args:
        data: Complete v3 file bytes.
        target_user_id: User to remove.

    Returns:
        Updated v3 file bytes with the user removed.

    Raises:
        ValueError: if user not found or would leave no users
    """
    header = unpack_v3_header(data)

    users = header.get("users", [])
    new_users = [u for u in users if u["user_id"] != target_user_id]

    if len(new_users) == len(users):
        raise ValueError(f"User '{target_user_id}' not found")

    if len(new_users) == 0 and not header.get("master"):
        raise ValueError(
            "Cannot remove last user without a master unlock configured"
        )

    header["users"] = new_users

    # Rebuild file with updated header, same payload bytes
    sep_idx = data.find(V3_SEPARATOR)
    payload_section = data[sep_idx:]

    header_json = json.dumps(header, ensure_ascii=False).encode("utf-8")
    header_len = struct.pack(">I", len(header_json))
    return header_len + header_json + payload_section


def v3_get_data_key_with_pin(data: bytes, user_id: str, pin: str) -> bytes:
    """Authenticate a user and return the unwrapped data key.

    Useful when the caller needs the data key for add_user or other operations.

    Returns:
        32-byte data key
    """
    header = unpack_v3_header(data)

    user_slot = None
    for u in header.get("users", []):
        if u["user_id"] == user_id:
            user_slot = u
            break
    if user_slot is None:
        raise ValueError(f"User '{user_id}' not found in v3 header")

    pin_salt = bytes.fromhex(user_slot["pin_salt"])
    pin_key, _ = derive_pin_key(pin, pin_salt)
    wrapped = bytes.fromhex(user_slot["encrypted_data_key"])
    return decrypt_data_key(wrapped, pin_key)


def v3_get_data_key_with_master(data: bytes, master_code: str) -> bytes:
    """Authenticate with master code and return the unwrapped data key.

    Returns:
        32-byte data key
    """
    header = unpack_v3_header(data)

    master_block = header.get("master")
    if not master_block:
        raise ValueError("No master unlock configured for this file")

    master_salt = bytes.fromhex(master_block["salt"])
    master_key, _ = derive_key(master_code, master_salt)
    wrapped = bytes.fromhex(master_block["encrypted_data_key"])
    return decrypt_data_key(wrapped, master_key)
