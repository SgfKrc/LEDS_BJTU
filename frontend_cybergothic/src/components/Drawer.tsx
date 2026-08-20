/**
 * 详情抽屉 — 键盘可关闭（Escape）、焦点收敛、背景滚动锁定（§5.5）。
 *
 * 触屏不依赖 hover：由行内「详情」按钮显式打开。
 */

import { useEffect, useRef, type ReactNode } from 'react';
import { X } from 'lucide-react';

interface DrawerProps {
  open: boolean;
  title: string;
  /** 标题上方的小号标签。 */
  tag?: string;
  onClose: () => void;
  children: ReactNode;
  /** 底部操作区。 */
  footer?: ReactNode;
}

export function Drawer({ open, title, tag, onClose, children, footer }: DrawerProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;
    restoreFocusRef.current = document.activeElement as HTMLElement | null;
    document.body.dataset.scrollLocked = 'true';
    closeRef.current?.focus();
    return () => {
      delete document.body.dataset.scrollLocked;
      // 关闭后把焦点还给触发按钮，键盘用户不会丢失位置。
      restoreFocusRef.current?.focus?.();
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key !== 'Tab') return;

      const panel = panelRef.current;
      if (!panel) return;
      const focusables = panel.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled])',
      );
      if (focusables.length === 0) return;
      const first = focusables[0]!;
      const last = focusables[focusables.length - 1]!;

      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };

    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="drawer" role="dialog" aria-modal="true" aria-label={title}>
      <button
        type="button"
        className="drawer__scrim"
        onClick={onClose}
        aria-label="关闭详情"
        tabIndex={-1}
      />
      <div className="drawer__panel" ref={panelRef}>
        <header className="drawer__head">
          <div>
            {tag ? <span className="mono-label">{tag}</span> : null}
            <h2 className="drawer__title">{title}</h2>
          </div>
          <button type="button" className="iconbtn" onClick={onClose} ref={closeRef}>
            <X size={18} strokeWidth={2.25} />
            <span className="sr-only">关闭详情</span>
          </button>
        </header>
        <div className="drawer__body">{children}</div>
        {footer ? <footer className="drawer__foot">{footer}</footer> : null}
      </div>
    </div>
  );
}
