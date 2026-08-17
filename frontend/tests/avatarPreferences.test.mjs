import assert from 'node:assert/strict';
import test from 'node:test';

import {
  USER_AVATAR_MAX_BYTES,
  isStoredUserAvatar,
  saveStoredUserAvatar,
  validateUserAvatarFile,
} from '../src/avatarPreferences.js';

const PNG_DATA_URL = 'data:image/png;base64,AA==';

test('user avatar preferences only accept bounded local raster data', () => {
  assert.equal(isStoredUserAvatar(PNG_DATA_URL), true);
  assert.equal(isStoredUserAvatar('data:image/svg+xml;base64,AA=='), false);
  assert.equal(isStoredUserAvatar('https://example.test/avatar.png'), false);
  assert.equal(isStoredUserAvatar(`data:image/png;base64,${'a'.repeat(USER_AVATAR_MAX_BYTES * 2)}`), false);
});

test('user avatar preferences persist only through the supplied local storage', () => {
  const values = new Map();
  const storage = {
    getItem: (key) => values.get(key) || null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  };
  assert.equal(saveStoredUserAvatar(PNG_DATA_URL, storage), PNG_DATA_URL);
  assert.equal([...values.values()][0], PNG_DATA_URL);
  assert.equal(saveStoredUserAvatar(null, storage), null);
  assert.equal(values.size, 0);
});

test('user avatar file validation rejects unsupported and oversized inputs', () => {
  assert.equal(validateUserAvatarFile({ type: 'image/png', size: 12 }), null);
  assert.match(validateUserAvatarFile({ type: 'image/svg+xml', size: 12 }), /PNG、JPEG 或 WebP/);
  assert.match(validateUserAvatarFile({ type: 'image/png', size: USER_AVATAR_MAX_BYTES + 1 }), /1 MiB/);
});
