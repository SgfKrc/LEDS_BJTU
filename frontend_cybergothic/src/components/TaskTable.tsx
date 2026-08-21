/**
 * 任务表格 — 覆盖加载 / 空 / 错误 / 成功 / 批量操作五种状态（§5.3）。
 *
 * 表头 sticky；窄屏由 CSS 转为纵向条目（每个单元格用 data-label 显示列名）。
 * 行内的「详情」是真实 button，不依赖 div:hover（§5.5）。
 */

import type { ReactNode } from 'react';
import { EmptyState, SkeletonRows } from './EmptyState';
import { CommandButton } from './CommandButton';
import type { ApiErrorKind, LoadState } from '../data/types';

export interface Column<T> {
  key: string;
  header: string;
  /** 单元格渲染；返回文本或元素。 */
  render: (row: T) => ReactNode;
  /** 等宽数字列右对齐。 */
  numeric?: boolean;
  /** 窄屏隐藏次要列。 */
  secondary?: boolean;
}

interface TaskTableProps<T> {
  caption: string;
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  state: LoadState;
  error?: string;
  errorKind?: ApiErrorKind | null;
  errorStatus?: number | null;
  /** 空数据文案。 */
  emptyTitle?: string;
  emptyDescription?: string;
  /** 重试回调；错误态显示按钮。 */
  onRetry?: () => void;
  /** 打开详情。 */
  onOpenRow?: (row: T) => void;
  /** 已选中的行 key；提供时显示复选框与批量操作条。 */
  selected?: Set<string>;
  onToggleSelect?: (key: string) => void;
  onToggleSelectAll?: () => void;
  /** 批量操作区，仅在有选中项时显示。 */
  bulkActions?: ReactNode;
  /** 行是否可选；不可选的行不显示复选框。 */
  isSelectable?: (row: T) => boolean;
}

export function TaskTable<T>({
  caption,
  columns,
  rows,
  rowKey,
  state,
  error = '',
  errorKind = null,
  errorStatus = null,
  emptyTitle = '暂无数据',
  emptyDescription,
  onRetry,
  onOpenRow,
  selected,
  onToggleSelect,
  onToggleSelectAll,
  bulkActions,
  isSelectable,
}: TaskTableProps<T>) {
  const selectable = Boolean(selected && onToggleSelect);
  const selectedCount = selected?.size ?? 0;

  if (state === 'error') {
    return (
      <EmptyState
        kind="error"
        title="数据加载失败"
        description="请检查后端是否运行，或稍后重试。"
        detail={error}
        errorKind={errorKind}
        errorStatus={errorStatus}
        {...(onRetry
          ? { action: <CommandButton variant="ghost" size="sm" onClick={onRetry}>重试</CommandButton> }
          : {})}
      />
    );
  }

  if (state === 'loading' && rows.length === 0) {
    return <SkeletonRows rows={4} columns={Math.min(columns.length, 4)} />;
  }

  if (state !== 'loading' && rows.length === 0) {
    return (
      <EmptyState
        kind="empty"
        title={emptyTitle}
        {...(emptyDescription ? { description: emptyDescription } : {})}
        {...(onRetry
          ? { action: <CommandButton variant="ghost" size="sm" onClick={onRetry}>刷新</CommandButton> }
          : {})}
      />
    );
  }

  const selectableRows = isSelectable ? rows.filter(isSelectable) : rows;
  const allSelected =
    selectableRows.length > 0 &&
    selectableRows.every((row) => selected?.has(rowKey(row)));

  return (
    <div className="ttable__wrap">
      {selectable && selectedCount > 0 ? (
        <div className="ttable__bulk" role="region" aria-label="批量操作">
          <span className="mono-label">已选 {selectedCount} 项</span>
          <div className="ttable__bulk-actions">{bulkActions}</div>
        </div>
      ) : null}

      <div className="ttable__scroll">
        <table className="ttable">
          <caption className="sr-only">{caption}</caption>
          <thead>
            <tr>
              {selectable ? (
                <th scope="col" className="ttable__select-col">
                  <label className="checkbox">
                    <input
                      type="checkbox"
                      checked={allSelected}
                      onChange={() => onToggleSelectAll?.()}
                      aria-label="全选"
                    />
                    <span className="checkbox__box" aria-hidden="true" />
                  </label>
                </th>
              ) : null}
              {columns.map((col) => (
                <th
                  key={col.key}
                  scope="col"
                  className={[
                    col.numeric ? 'is-numeric' : '',
                    col.secondary ? 'is-secondary' : '',
                  ]
                    .filter(Boolean)
                    .join(' ')}
                >
                  {col.header}
                </th>
              ))}
              {onOpenRow ? <th scope="col" className="ttable__action-col">操作</th> : null}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const key = rowKey(row);
              const canSelect = isSelectable ? isSelectable(row) : true;
              return (
                <tr key={key} data-selected={selected?.has(key) ? 'true' : undefined}>
                  {selectable ? (
                    <td className="ttable__select-col">
                      {canSelect ? (
                        <label className="checkbox">
                          <input
                            type="checkbox"
                            checked={selected?.has(key) ?? false}
                            onChange={() => onToggleSelect?.(key)}
                            aria-label={`选择 ${key}`}
                          />
                          <span className="checkbox__box" aria-hidden="true" />
                        </label>
                      ) : null}
                    </td>
                  ) : null}
                  {columns.map((col) => (
                    <td
                      key={col.key}
                      data-label={col.header}
                      className={[
                        col.numeric ? 'is-numeric' : '',
                        col.secondary ? 'is-secondary' : '',
                      ]
                        .filter(Boolean)
                        .join(' ')}
                    >
                      {col.render(row)}
                    </td>
                  ))}
                  {onOpenRow ? (
                    <td className="ttable__action-col" data-label="操作">
                      <button
                        type="button"
                        className="ttable__open"
                        onClick={() => onOpenRow(row)}
                      >
                        详情
                      </button>
                    </td>
                  ) : null}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
