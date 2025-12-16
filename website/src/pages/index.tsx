import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className="text--center">
          <h1 className="hero__title">Introduction to Physical AI & Humanoid Robotics</h1>
          <p className="hero__subtitle">A comprehensive textbook on the cutting-edge intersection of artificial intelligence and robotics</p>
          <div className={styles.buttons}>
            <Link
              className="button button--secondary button--lg"
              to="/introduction">
              Start Reading - 5 min ⏱️
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}

function HeroSection() {
  return (
    <section className={styles.heroSection}>
      <div className="container">
        <div className="row">
          <div className="col col--6">
            <div className={styles.heroContent}>
              <h2>Physical AI & Humanoid Robotics</h2>
              <p>
                Welcome to the fascinating world of Physical AI and Humanoid Robotics!
                This textbook provides a comprehensive exploration of the intersection between
                artificial intelligence and physical robots, with a particular focus on humanoid
                systems that are designed to interact with humans and operate in human environments.
              </p>
              <div className={styles.keyPoints}>
                <div className={styles.keyPoint}>
                  <h3>Physical AI</h3>
                  <p>AI systems that operate in and interact with the physical world</p>
                </div>
                <div className={styles.keyPoint}>
                  <h3>Humanoid Robotics</h3>
                  <p>Robots designed with human-like form and capabilities</p>
                </div>
                <div className={styles.keyPoint}>
                  <h3>Real-time Interaction</h3>
                  <p>Systems that operate safely in dynamic physical environments</p>
                </div>
              </div>
            </div>
          </div>
          <div className="col col--6">
            <div className={styles.heroImage}>
              <img
                src="/img/undraw_docusaurus_tree.svg"
                alt="Humanoid Robotics Illustration"
                className={styles.heroImageContent}
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Introduction to Physical AI & Humanoid Robotics`}
      description="Physical AI & Humanoid Robotics Textbook - Learn about the cutting-edge intersection of artificial intelligence and robotics">
      <HomepageHeader />
      <HeroSection />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}