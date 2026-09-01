-- Enables RLS (with no policies, i.e. deny-by-default for the anon/
-- authenticated API roles) on every table in the public schema that
-- doesn't already have it. Run after reviewing check_rls.sql output.
--
-- Run with:
--   psql "$DATABASE_URL" -f enable_rls_all.sql
-- or paste into the Supabase SQL editor.
--
-- NOTE: this only protects tables from Supabase's auto-generated REST/
-- GraphQL API (the anon/authenticated roles). Any table your app reads
-- via a direct DATABASE_URL connection (like diabetes_risk) keeps
-- working, since that connection isn't subject to these policies.
-- If any table DOES need to be readable via the public API, don't run
-- this blindly for it -- add a scoped policy instead (see enable_rls.sql
-- for an example).

DO $$
DECLARE
    tbl RECORD;
BEGIN
    FOR tbl IN
        SELECT c.relname
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relkind = 'r'
          AND NOT c.relrowsecurity
    LOOP
        EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY;', tbl.relname);
        RAISE NOTICE 'Enabled RLS on public.%', tbl.relname;
    END LOOP;
END $$;
