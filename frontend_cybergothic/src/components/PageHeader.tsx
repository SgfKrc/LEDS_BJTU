/**
 * 页面标题区 — 标题从左侧淡入 + 切角高亮条展开，只播放一次（§5.4）。
 */

import type { ReactNode } from 'react';

interface PageHeaderProps {
  /** 小号英文标签，仅作品牌气质（§4.1）。 */
  tag: string;
  title: string;
  description?: string;
  /** 右侧操作区。 */
  actions?: ReactNode;
}

export function PageHeader({ tag, title, description, actions }: PageHeaderProps) {
  return (
    <header className="pagehead" data-enter>
      <div className="pagehead__main">
        <span className="pagehead__tag mono-label">{tag}</span>
        <h1 className="pagehead__title">
          {title}
          <span className="pagehead__rule" aria-hidden="true" />
        </h1>
        {description ? <p className="pagehead__desc lede">{description}</p> : null}
      </div>
      {actions ? <div className="pagehead__actions">{actions}</div> : null}
    </header>
  );
}

/** 区块标题 — 整宽色带内的次级标题，不再包一层卡片（§4.1 第 4 条）。 */
export function SectionHead({
  title,
  hint,
  actions,
  id,
}: {
  title: string;
  hint?: string;
  actions?: ReactNode;
  id?: string;
}) {
  return (
    <div className="sectionhead">
      <div>
        <h2 className="section-title" {...(id ? { id } : {})}>
          {title}
        </h2>
        {hint ? <p className="sectionhead__hint">{hint}</p> : null}
      </div>
      {actions ? <div className="sectionhead__actions">{actions}</div> : null}
    </div>
  );
}
