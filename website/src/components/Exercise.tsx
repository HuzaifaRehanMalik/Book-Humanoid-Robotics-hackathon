import React from 'react';
import clsx from 'clsx';
import styles from './Exercise.module.css';

type ExerciseProps = {
  children: React.ReactNode;
  title?: string;
  difficulty?: 'beginner' | 'intermediate' | 'advanced';
  className?: string;
};

const Exercise: React.FC<ExerciseProps> = ({ children, title, difficulty = 'intermediate', className }) => {
  const difficultyClass = styles[`difficulty-${difficulty}`];

  return (
    <div className={clsx(styles.exercise, className, difficultyClass)}>
      <div className={styles.header}>
        <h4 className={styles.title}>
          {title || 'Exercise'}
          <span className={styles.difficulty}>{difficulty.charAt(0).toUpperCase() + difficulty.slice(1)}</span>
        </h4>
      </div>
      <div className={styles.content}>{children}</div>
    </div>
  );
};

export default Exercise;