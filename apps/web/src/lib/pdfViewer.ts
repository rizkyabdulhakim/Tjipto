export function fitWidthScale(pageWidth: number, availableWidth: number, zoom: number) {
  return pageWidth > 0 && availableWidth > 0 ? (availableWidth / pageWidth) * zoom : 1;
}

export function canvasBackingStore(width: number, height: number, devicePixelRatio: number) {
  const ratio = Number.isFinite(devicePixelRatio) && devicePixelRatio > 0 ? devicePixelRatio : 1;
  return {
    width: Math.max(1, Math.ceil(width * ratio)),
    height: Math.max(1, Math.ceil(height * ratio)),
    ratio,
  };
}

export function visiblePageWindow(visiblePages: Iterable<number>, pageCount: number) {
  const pages = new Set<number>();
  for (const page of visiblePages) {
    for (let candidate = page - 1; candidate <= page + 1; candidate += 1) {
      if (candidate >= 1 && candidate <= pageCount) pages.add(candidate);
    }
  }
  return pages;
}

export function isRenderCancellation(error: unknown) {
  return error instanceof Error && error.name === "RenderingCancelledException";
}

export interface CancelableRenderTask {
  cancel(): void;
}

export class RenderTaskOwner<T extends CancelableRenderTask> {
  private task: T | null = null;

  replace(task: T) {
    this.cancel();
    this.task = task;
    return task;
  }

  isCurrent(task: T) {
    return this.task === task;
  }

  finish(task: T) {
    if (this.task === task) this.task = null;
  }

  cancel() {
    this.task?.cancel();
    this.task = null;
  }
}
