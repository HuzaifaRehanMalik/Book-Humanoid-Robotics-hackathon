import React from 'react';
import clsx from 'clsx';
import styles from './Figure.module.css';

type FigureProps = {
  children: React.ReactNode;
  caption?: string;
  className?: string;
  align?: 'left' | 'center' | 'right';
};

const Figure: React.FC<FigureProps> = ({ children, caption, className, align = 'center' }) => {
  const alignmentClass = styles[`align-${align}`];

  return (
    <figure className={clsx(styles.figure, className, alignmentClass)}>
      <div className={styles.content}>{children}</div>
      {caption && <figcaption className={styles.caption}>{caption}</figcaption>}
    </figure>
  );
};

export default Figure;