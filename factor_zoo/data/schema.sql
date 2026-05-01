CREATE TABLE IF NOT EXISTS factors (
    id VARCHAR PRIMARY KEY,
    name VARCHAR,
    category VARCHAR,
    subcategory VARCHAR,
    authors VARCHAR,
    year INTEGER,
    journal VARCHAR,
    paper_url VARCHAR,
    sample_start DATE,
    sample_end DATE,
    n_stocks_avg INTEGER,
    description TEXT,
    -- precomputed stats (computed from factor_returns)
    ann_return FLOAT,
    ann_vol FLOAT,
    sharpe FLOAT,
    max_drawdown FLOAT,
    t_stat FLOAT,
    pre_pub_sharpe FLOAT,
    post_pub_sharpe FLOAT,
    has_short_history BOOLEAN DEFAULT FALSE,  -- TRUE if < 60 months
    source VARCHAR,                            -- 'osap' | 'french'
    paper_t_stat FLOAT                         -- reported t-stat from the paper (OSAP only)
);

CREATE TABLE IF NOT EXISTS factor_returns (
    factor_id VARCHAR,
    date DATE,
    ls_return FLOAT,   -- long-short monthly return as decimal (0.01 = 1%)
    PRIMARY KEY (factor_id, date)
);
