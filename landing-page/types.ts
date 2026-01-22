import React from 'react';

export interface CodeSnippet {
  title: string;
  description: string;
  code: string;
  filename: string;
}

export interface FeatureProps {
  title: string;
  description: string;
  icon?: React.ReactNode;
}