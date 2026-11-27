export declare const isPinned: (metrics: { scrollHeight: number; scrollTop: number; clientHeight: number }, threshold?: number) => boolean;
export declare function createAutoScroll(container: HTMLElement, options?: { threshold?: number; button?: HTMLElement | null }): { pin(): void; schedule(): void; readonly pinned: boolean };
