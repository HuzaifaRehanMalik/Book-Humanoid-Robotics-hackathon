import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import { useDoc } from '@docusaurus/plugin-content-docs/client';
import styles from './ChapterNavigation.module.css';

const ChapterNavigation: React.FC = () => {
  const { metadata } = useDoc();
  const { sidebarName, previous, next } = metadata;

  if (!previous && !next) {
    return null;
  }

  return (
    <nav className={clsx(styles.chapterNavigation, 'pagination-nav')}>
      <div className="container-fluid">
        <div className="row">
          {previous && (
            <div className="col col--6">
              <Link className={clsx(styles.navLink, 'pagination-nav__link', 'pagination-nav__link--prev')} to={previous.permalink}>
                <h5 className="pagination-nav__sublabel">Previous</h5>
                <span className={styles.navText}>
                  <i className="fa fa-arrow-left"></i> {previous.title}
                </span>
              </Link>
            </div>
          )}
          {next && (
            <div className={clsx('col', previous ? 'col--6' : 'col--12')}>
              <Link className={clsx(styles.navLink, 'pagination-nav__link', 'pagination-nav__link--next')} to={next.permalink}>
                <h5 className="pagination-nav__sublabel">Next</h5>
                <span className={styles.navText}>
                  {next.title} <i className="fa fa-arrow-right"></i>
                </span>
              </Link>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
};

export default ChapterNavigation;