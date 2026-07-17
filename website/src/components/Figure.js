import React from 'react';
import clsx from 'clsx';
import styles from './Figure.module.css';
const Figure = ({ children, caption, className, align = 'center' }) => {
    const alignmentClass = styles[`align-${align}`];
    return (<figure className={clsx(styles.figure, className, alignmentClass)}>
      <div className={styles.content}>{children}</div>
      {caption && <figcaption className={styles.caption}>{caption}</figcaption>}
    </figure>);
};
export default Figure;
