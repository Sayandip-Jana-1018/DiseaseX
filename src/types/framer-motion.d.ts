import * as React from 'react';

declare module 'framer-motion' {
  export interface MotionProps {
    initial?: any;
    animate?: any;
    exit?: any;
    variants?: any;
    transition?: any;
    whileHover?: any;
    whileTap?: any;
    whileFocus?: any;
    whileDrag?: any;
    whileInView?: any;
    style?: React.CSSProperties;
    className?: string;
    onClick?: () => void;
    onHoverStart?: () => void;
    onHoverEnd?: () => void;
    onTap?: () => void;
    onTapStart?: () => void;
    onTapCancel?: () => void;
    onDrag?: () => void;
    onDragStart?: () => void;
    onDragEnd?: () => void;
    onViewportEnter?: () => void;
    onViewportLeave?: () => void;
  }

  export interface HTMLMotionProps<T> extends React.HTMLAttributes<T>, MotionProps {
    ref?: React.RefObject<T>;
  }

  export interface ForwardRefComponent<T, P> extends React.ForwardRefExoticComponent<React.PropsWithoutRef<P> & React.RefAttributes<T>> {}

  export type MotionComponent<P = {}> = ForwardRefComponent<HTMLElement, P & MotionProps>;

  export interface Motion {
    div: MotionComponent<React.HTMLAttributes<HTMLDivElement>>;
    span: MotionComponent<React.HTMLAttributes<HTMLSpanElement>>;
    button: MotionComponent<React.ButtonHTMLAttributes<HTMLButtonElement>>;
    a: MotionComponent<React.AnchorHTMLAttributes<HTMLAnchorElement>>;
    ul: MotionComponent<React.HTMLAttributes<HTMLUListElement>>;
    ol: MotionComponent<React.HTMLAttributes<HTMLOListElement>>;
    li: MotionComponent<React.LiHTMLAttributes<HTMLLIElement>>;
    header: MotionComponent<React.HTMLAttributes<HTMLElement>>;
    footer: MotionComponent<React.HTMLAttributes<HTMLElement>>;
    nav: MotionComponent<React.HTMLAttributes<HTMLElement>>;
    section: MotionComponent<React.HTMLAttributes<HTMLElement>>;
    article: MotionComponent<React.HTMLAttributes<HTMLElement>>;
    aside: MotionComponent<React.HTMLAttributes<HTMLElement>>;
    main: MotionComponent<React.HTMLAttributes<HTMLElement>>;
    form: MotionComponent<React.FormHTMLAttributes<HTMLFormElement>>;
    input: MotionComponent<React.InputHTMLAttributes<HTMLInputElement>>;
    textarea: MotionComponent<React.TextareaHTMLAttributes<HTMLTextAreaElement>>;
    select: MotionComponent<React.SelectHTMLAttributes<HTMLSelectElement>>;
    h1: MotionComponent<React.HTMLAttributes<HTMLHeadingElement>>;
    h2: MotionComponent<React.HTMLAttributes<HTMLHeadingElement>>;
    h3: MotionComponent<React.HTMLAttributes<HTMLHeadingElement>>;
    h4: MotionComponent<React.HTMLAttributes<HTMLHeadingElement>>;
    h5: MotionComponent<React.HTMLAttributes<HTMLHeadingElement>>;
    h6: MotionComponent<React.HTMLAttributes<HTMLHeadingElement>>;
    p: MotionComponent<React.HTMLAttributes<HTMLParagraphElement>>;
    img: MotionComponent<React.ImgHTMLAttributes<HTMLImageElement>>;
    svg: MotionComponent<React.SVGAttributes<SVGSVGElement>>;
    path: MotionComponent<React.SVGAttributes<SVGPathElement>>;
  }

  export const motion: Motion;
}
