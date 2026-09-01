-- Fixes Supabase linter finding: "public.diabetes_risk is exposed via API
-- without RLS and contains potentially sensitive column(s): patient_id."
--
-- Run once against the Supabase project, e.g.:
--   psql "$DATABASE_URL" -f enable_rls.sql
-- or paste into the Supabase SQL editor.
--
-- Enabling RLS with no policies denies all access to the anon/authenticated
-- API roles used by Supabase's auto-generated REST/GraphQL API, while
-- app.py and upload_to_supabase.py keep working since they connect directly
-- via DATABASE_URL (a role that isn't subject to these API-facing policies).

ALTER TABLE public.diabetes_risk ENABLE ROW LEVEL SECURITY;

-- No policies are created: this blocks all anon/authenticated API access
-- by default, since this dataset is not meant to be queried over the
-- public API. If a future feature needs read access via the API, add a
-- narrow policy here, e.g.:
--
-- CREATE POLICY "authenticated_read_no_patient_id"
--   ON public.diabetes_risk
--   FOR SELECT
--   TO authenticated
--   USING (true);
--
-- (and expose a view without patient_id rather than the raw table).
