/**
 * 移动端菜单 — 打开时锁定背景滚动，支持 Escape 关闭，焦点收敛在面板内（§5.5）。
 */

import { useEffect, useRef } from 'react';
import { X } from 'lucide-react';
import { PRIMARY_NAV_IDS, ROUTES, routeHref, type RouteId } from '../routes';

interface MobileNavProps {
  open: boolean;
  currentId: RouteId;
  onClose: () => void;
  onNavigate: (id: RouteId) => void;
}

export function MobileNav({ open, currentId, onClose, onNavigate }: MobileNavProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  // 背景滚动锁定
  useEffect(() => {
    if (!open) return;
    document.body.dataset.scrollLocked = 'true';
    return () => {
      delete document.body.dataset.scrollLocked;
    };
  }, [open]);

  // Escape 关闭 + 简易焦点循环
  useEffect(() => {
    if (!open) return;

    closeRef.current?.focus();

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
        'a[href], button:not([disabled])',
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
    <div className="mobilenav" role="dialog" aria-modal="true" aria-label="站点菜单">
      <button
        type="button"
        className="mobilenav__scrim"
        onClick={onClose}
        aria-label="关闭菜单"
        tabIndex={-1}
      />
      <div className="mobilenav__panel" ref={panelRef}>
        <div className="mobilenav__head">
          <span className="mono-label">菜单</span>
          <button type="button" className="iconbtn" onClick={onClose} ref={closeRef}>
            <X size={18} strokeWidth={2.25} />
            <span className="sr-only">关闭菜单</span>
          </button>
        </div>
        <ul className="mobilenav__list">
          {ROUTES.map((route) => {
            const Icon = route.icon;
            const isActive = route.id === currentId;
            const isPrimary = PRIMARY_NAV_IDS.includes(route.id);
            return (
              <li key={route.id} className={isPrimary ? 'mobilenav__item mobilenav__item--primary' : 'mobilenav__item'}>
                <a
                  className="mobilenav__link"
                  href={routeHref(route.id)}
                  aria-current={isActive ? 'page' : undefined}
                  data-active={isActive ? 'true' : undefined}
                  onClick={(e) => {
                    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
                    e.preventDefault();
                    onNavigate(route.id);
                    onClose();
                  }}
                >
                  <span className="mobilenav__icon" aria-hidden="true">
                    <Icon size={17} strokeWidth={2.25} />
                  </span>
                  <span className="mobilenav__text">
                    <span className="mobilenav__label">{route.label}</span>
                    <span className="mobilenav__desc">{route.description}</span>
                  </span>
                </a>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
