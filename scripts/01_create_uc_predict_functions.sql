-- =============================================================================
-- 01_create_uc_predict_functions.sql
-- J&J HRD 2030 — UC-registered SQL tools for predictive hiring
--
-- Creates two Unity Catalog SQL functions that wrap ai_query() to call the
-- hr-predictive-hiring-endpoint model serving endpoint:
--
--   1. predict_hiring_score(8 feature ints) → STRING
--      Core scoring function — takes feature scores, returns raw model prediction.
--
--   2. predict_hiring_score_by_id(candidate_id) → STRING
--      Convenience wrapper — looks up candidate from Delta table, delegates to (1).
--      Used by the agent as a single-argument tool.
--
-- Run against: bx4.hrd_2030  |  Warehouse: SQL Warehouse (not Serverless DLT)
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. Core scoring function — takes the 8 feature scores directly
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION bx4.hrd_2030.predict_hiring_score(
  education_score          INT  COMMENT 'Education qualification score (0–100)',
  experience_score         INT  COMMENT 'Total years experience score (0–100)',
  leadership_score         INT  COMMENT 'Leadership experience score (0–100)',
  certification_score      INT  COMMENT 'Professional certifications score (0–100)',
  skills_match_score       INT  COMMENT 'Job skills match score (0–100)',
  industry_relevance_score INT  COMMENT 'Industry relevance score (0–100)',
  interview_score          INT  COMMENT 'Interview performance score (0–100)',
  culture_fit              INT  COMMENT 'Culture fit score (0–100)'
)
RETURNS STRING
COMMENT 'Call the hr-predictive-hiring-endpoint ML model via ai_query to predict hire/no-hire. Returns JSON with the model prediction (0=no hire, 1=hire).'
RETURN (
  SELECT CAST(
    ai_query(
      'hr-predictive-hiring-endpoint',
      named_struct(
        'dataframe_records', array(
          named_struct(
            'education_score',          predict_hiring_score.education_score,
            'experience_score',         predict_hiring_score.experience_score,
            'leadership_score',         predict_hiring_score.leadership_score,
            'certification_score',      predict_hiring_score.certification_score,
            'skills_match_score',       predict_hiring_score.skills_match_score,
            'industry_relevance_score', predict_hiring_score.industry_relevance_score,
            'interview_score',          predict_hiring_score.interview_score,
            'culture_fit',              predict_hiring_score.culture_fit
          )
        )
      )
    ) AS STRING
  )
);

-- ---------------------------------------------------------------------------
-- 2. Candidate-ID wrapper — looks up from Delta, delegates to function (1)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION bx4.hrd_2030.predict_hiring_score_by_id(
  candidate_id STRING COMMENT 'Candidate ID, e.g. C001 or C011'
)
RETURNS STRING
COMMENT 'Predict hire/no-hire for a candidate by their ID. Fetches the 8 feature scores from the candidates table and calls the ML model via ai_query. For new candidates (C011–C020) missing interview scores, returns a message requesting the scores.'
RETURN (
  SELECT
    CASE
      WHEN education_score IS NULL OR experience_score IS NULL
           OR leadership_score IS NULL OR certification_score IS NULL
           OR industry_relevance_score IS NULL
      THEN CONCAT('Candidate ', predict_hiring_score_by_id.candidate_id,
                  ' is missing required feature scores. Check candidates table.')

      WHEN skills_match_score IS NULL OR interview_score IS NULL OR culture_fit IS NULL
      THEN CONCAT('Candidate ', predict_hiring_score_by_id.candidate_id,
                  ' is a new candidate (', COALESCE(first_name, ''), ' ', COALESCE(last_name, ''), ')',
                  ' — missing post-interview scores: ',
                  CASE WHEN skills_match_score IS NULL THEN 'skills_match_score ' ELSE '' END,
                  CASE WHEN interview_score    IS NULL THEN 'interview_score '    ELSE '' END,
                  CASE WHEN culture_fit        IS NULL THEN 'culture_fit '        ELSE '' END,
                  '(0–100 each). Please supply these after the interview.')

      ELSE bx4.hrd_2030.predict_hiring_score(
        CAST(education_score          AS INT),
        CAST(experience_score         AS INT),
        CAST(leadership_score         AS INT),
        CAST(certification_score      AS INT),
        CAST(skills_match_score       AS INT),
        CAST(industry_relevance_score AS INT),
        CAST(interview_score          AS INT),
        CAST(culture_fit              AS INT)
      )
    END
  FROM bx4.hrd_2030.candidates
  WHERE UPPER(candidates.candidate_id) = UPPER(predict_hiring_score_by_id.candidate_id)
  LIMIT 1
);
