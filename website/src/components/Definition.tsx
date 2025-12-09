import React from 'react';
import clsx from 'clsx';
import styles from './Definition.module.css';

type DefinitionProps = {
  children: React.ReactNode;
  term?: string;
  className?: string;
};

const Definition: React.FC<DefinitionProps> = ({ children, term, className }) => {
  return (
    <div className={clsx(styles.definition, className)}>
      {term && <h4 className={styles.term}>{term}</h4>}
      <div className={styles.content}>{children}</div>
    </div>
  );
};

export default Definition;