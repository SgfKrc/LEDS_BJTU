/**
 * 主操作按钮 — 真实 button/a 元素，图标 + 短文本（§5.1、§5.5）。
 */

import type { ReactNode } from 'react';
import type { LucideIcon } from 'lucide-react';

type Variant = 'primary' | 'ghost' | 'danger';

interface BaseProps {
  children: ReactNode;
  /** lucide 图标组件，可选。 */
  icon?: LucideIcon;
  variant?: Variant;
  size?: 'sm' | 'md';
  /** 进行中：禁用点击并显示进度文案。 */
  busy?: boolean;
  className?: string;
}

interface ButtonProps extends BaseProps {
  /** type="submit" 时可省略，由表单的 onSubmit 处理。 */
  onClick?: () => void;
  href?: never;
  disabled?: boolean;
  type?: 'button' | 'submit';
  /** 无障碍标签，当 children 只有图标时必填。 */
  ariaLabel?: string;
}

interface LinkProps extends BaseProps {
  href: string;
  onClick?: never;
  external?: boolean;
  ariaLabel?: string;
}

export function CommandButton(props: ButtonProps | LinkProps) {
  const {
    children,
    icon: Icon,
    variant = 'primary',
    size = 'md',
    busy = false,
    className = '',
  } = props;

  const classes = [
    'cbtn',
    `cbtn--${variant}`,
    `cbtn--${size}`,
    busy ? 'is-busy' : '',
    className,
  ]
    .filter(Boolean)
    .join(' ');

  const inner = (
    <>
      {Icon ? (
        <span className="cbtn__icon" aria-hidden="true">
          <Icon size={size === 'sm' ? 14 : 16} strokeWidth={2.25} />
        </span>
      ) : null}
      <span className="cbtn__text">{children}</span>
    </>
  );

  if ('href' in props && props.href) {
    const isExternal = props.external ?? /^https?:/i.test(props.href);
    return (
      <a
        className={classes}
        href={props.href}
        {...(props.ariaLabel ? { 'aria-label': props.ariaLabel } : {})}
        {...(isExternal ? { target: '_blank', rel: 'noreferrer noopener' } : {})}
      >
        {inner}
      </a>
    );
  }

  const { onClick, disabled = false, type = 'button', ariaLabel } = props as ButtonProps;

  return (
    <button
      className={classes}
      type={type}
      {...(onClick ? { onClick } : {})}
      disabled={disabled || busy}
      {...(busy ? { 'aria-busy': true } : {})}
      {...(ariaLabel ? { 'aria-label': ariaLabel } : {})}
    >
      {inner}
    </button>
  );
}
