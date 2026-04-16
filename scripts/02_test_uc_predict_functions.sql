-- =============================================================================
-- 02_test_uc_predict_functions.sql
-- J&J HRD 2030 — Test the UC SQL prediction functions
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Test 1: Call ai_query() directly (matches the user's example format)
-- Scores for 5 candidates inline — no table lookup
-- ---------------------------------------------------------------------------
SELECT ai_query(
  'hr-predictive-hiring-endpoint',
  request => named_struct(
    'dataframe_split', named_struct(
      'columns', array(
        'education_score', 'experience_score', 'leadership_score',
        'certification_score', 'skills_match_score', 'industry_relevance_score',
        'interview_score', 'culture_fit'
      ),
      'data', array(
        array(85, 65, 55, 75, 60, 55, 65, 70),   -- candidate A
        array(75, 85, 80, 90, 85, 90, 82, 85),   -- candidate B
        array(60, 70, 45, 65, 58, 45, 62, 65),   -- candidate C
        array(85, 95, 92, 95, 95, 95, 90, 92),   -- candidate D (high scorer)
        array(85, 90, 85, 95, 92, 95, 88, 88)    -- candidate E
      )
    )
  )
) AS bulk_predictions;

-- ---------------------------------------------------------------------------
-- Test 2: Core scoring function with explicit feature scores
-- ---------------------------------------------------------------------------
SELECT bx4.hrd_2030.predict_hiring_score(
  85, 90, 85, 95, 92, 95, 88, 88   -- Sarah Chen-like scores
) AS prediction_sarah_chen;

SELECT bx4.hrd_2030.predict_hiring_score(
  70, 40, 30, 45, 35, 40, 42, 45  -- low scorer
) AS prediction_low_scorer;

-- ---------------------------------------------------------------------------
-- Test 3: Candidate-ID wrapper for existing candidates (C001–C010)
-- ---------------------------------------------------------------------------
SELECT bx4.hrd_2030.predict_hiring_score_by_id('C001') AS c001_sarah_chen;
SELECT bx4.hrd_2030.predict_hiring_score_by_id('C004') AS c004_david_kim;
SELECT bx4.hrd_2030.predict_hiring_score_by_id('C006') AS c006_robert_johnson;

-- ---------------------------------------------------------------------------
-- Test 4: New candidates (C011–C020) — should return "missing scores" message
-- ---------------------------------------------------------------------------
SELECT bx4.hrd_2030.predict_hiring_score_by_id('C011') AS c011_new_candidate;

-- ---------------------------------------------------------------------------
-- Test 5: Score all C001–C010 candidates in one query
-- ---------------------------------------------------------------------------
SELECT
  candidate_id,
  first_name,
  last_name,
  job_id,
  bx4.hrd_2030.predict_hiring_score_by_id(candidate_id) AS ml_prediction,
  hired AS actual_label
FROM bx4.hrd_2030.candidates
WHERE candidate_id BETWEEN 'C001' AND 'C010'
ORDER BY candidate_id;
