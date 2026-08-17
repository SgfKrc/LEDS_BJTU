export const USER_AVATAR_STORAGE_KEY = 'qlh-user-avatar-v1';
export const USER_AVATAR_MAX_BYTES = 1024 * 1024;

const USER_AVATAR_DATA_URL = /^data:image\/(?:png|jpeg|webp);base64,[a-z0-9+/=\s]+$/i;

export function isStoredUserAvatar(value) {
  return typeof value === 'string'
    && value.length <= Math.ceil(USER_AVATAR_MAX_BYTES * 4 / 3) + 128
    && USER_AVATAR_DATA_URL.test(value);
}

export function readStoredUserAvatar(storage = globalThis.localStorage) {
  try {
    const value = storage?.getItem(USER_AVATAR_STORAGE_KEY);
    return isStoredUserAvatar(value) ? value : null;
  } catch (_) {
    return null;
  }
}

export function saveStoredUserAvatar(value, storage = globalThis.localStorage) {
  if (!value) {
    storage?.removeItem(USER_AVATAR_STORAGE_KEY);
    return null;
  }
  if (!isStoredUserAvatar(value)) {
    throw new Error('头像数据无效或超过 1 MiB 上限');
  }
  storage?.setItem(USER_AVATAR_STORAGE_KEY, value);
  return value;
}

export function validateUserAvatarFile(file) {
  if (!file) return '请选择图片文件';
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) {
    return '头像仅支持 PNG、JPEG 或 WebP';
  }
  if (file.size > USER_AVATAR_MAX_BYTES) {
    return '头像不能超过 1 MiB';
  }
  return null;
}

export function readUserAvatarFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('无法读取头像文件'));
    reader.onload = () => {
      const value = String(reader.result || '');
      if (!isStoredUserAvatar(value)) {
        reject(new Error('头像文件内容无效或超过 1 MiB 上限'));
        return;
      }
      resolve(value);
    };
    reader.readAsDataURL(file);
  });
}
